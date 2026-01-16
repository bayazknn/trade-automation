"""
Dual-CNN LSTM Model Module

Implements the Dual-CNN LSTM architecture for binary classification of trading signals.

Architecture:
1. CNN1 Branch: Processes binary indicator signals (pure 0/1 entry/exit signals)
2. CNN2 Branch: Processes technical indicators + OHLCV (scaled continuous values)
3. Concatenation: Combines outputs from both CNN branches
4. LSTM: Captures temporal dependencies from combined features
5. Classifier: Binary prediction (hold=0, trade=1)

Key Design Decisions:
- GELU activations instead of ReLU (smoother gradients, avoids dead neurons)
- BatchNorm1d in CNN branches, LayerNorm in classifier
- Light dropout only before classifier, none in CNN branches
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DualModelConfig:
    """Configuration for Dual-CNN LSTM model."""

    # Required fields (no defaults) - must come first
    cnn1_input_features: int  # Number of binary input features
    cnn2_input_features: int  # Number of technical + OHLCV input features

    # CNN1 Branch (Binary features)
    cnn1_kernel_size: int = 5  # Conv1d kernel size
    cnn1_num_channels: int = 64  # Output channels per conv layer
    cnn1_num_layers: int = 2  # Number of conv layers

    # CNN2 Branch (Technical + OHLCV features)
    cnn2_kernel_size: int = 5  # Conv1d kernel size
    cnn2_num_channels: int = 64  # Output channels per conv layer
    cnn2_num_layers: int = 2  # Number of conv layers

    # Fusion Layer (between CNN concat and LSTM)
    fusion_hidden_size: int = 128  # Output size of fusion layer (0 = no fusion layer)
    fusion_dropout: float = 0.1  # Dropout after fusion linear layer

    # LSTM
    lstm_hidden_size: int = 128  # LSTM hidden state size
    lstm_num_layers: int = 2  # Number of LSTM layers
    lstm_dropout: float = 0.1  # Dropout between LSTM layers
    lstm_bidirectional: bool = False  # Use bidirectional LSTM

    # Classifier
    classifier_hidden_size: int = 64  # Hidden layer size (0 = single layer)
    classifier_dropout: float = 0.1  # Dropout before final layer

    # Output
    num_classes: int = 2  # Binary: hold=0, trade=1

    # Sequence
    input_seq_length: int = 16  # Length of input sequences


class CNNBranch(nn.Module):
    """
    CNN branch for feature extraction.

    Uses Conv1d layers with BatchNorm and GELU activations.
    Preserves sequence length using 'same' padding.
    """

    def __init__(
        self,
        input_features: int,
        num_channels: int,
        kernel_size: int,
        num_layers: int = 2
    ):
        """
        Initialize CNN branch.

        Parameters
        ----------
        input_features : int
            Number of input features (channels)
        num_channels : int
            Number of output channels for conv layers
        kernel_size : int
            Kernel size for Conv1d layers
        num_layers : int
            Number of convolutional layers
        """
        super().__init__()

        layers = []

        # First conv layer: input_features -> num_channels
        # Use padding='same' to always preserve sequence length regardless of kernel size
        layers.append(nn.Conv1d(input_features, num_channels, kernel_size, padding='same'))
        layers.append(nn.BatchNorm1d(num_channels))
        layers.append(nn.GELU())

        # Additional conv layers: num_channels -> num_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(num_channels, num_channels, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(num_channels))
            layers.append(nn.GELU())

        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = num_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, input_features)

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, seq_len, num_channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_features, seq_len)
        x = self.conv_layers(x)  # (batch, num_channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, num_channels)
        return x


class DualCNNLSTMPredictor(nn.Module):
    """
    Dual-CNN LSTM model for binary trading signal prediction.

    Architecture:
    - Two parallel CNN branches for binary and technical features
    - Feature concatenation after CNN processing
    - LSTM encoder for temporal dependencies
    - Classification head for binary prediction

    Input:
    - binary_features: (batch, seq_len, n_binary) - pure 0/1 signals
    - technical_features: (batch, seq_len, n_technical) - scaled indicators + OHLCV

    Output:
    - logits: (batch, num_classes) = (batch, 2) for hold/trade prediction

    Examples
    --------
    >>> config = DualModelConfig(
    ...     cnn1_input_features=114,
    ...     cnn2_input_features=81
    ... )
    >>> model = DualCNNLSTMPredictor(config)
    >>> binary = torch.randn(32, 16, 114)
    >>> technical = torch.randn(32, 16, 81)
    >>> output = model(binary, technical)
    >>> print(output.shape)
    torch.Size([32, 2])
    """

    def __init__(self, config: DualModelConfig):
        """
        Initialize the model.

        Parameters
        ----------
        config : DualModelConfig
            Model configuration
        """
        super().__init__()
        self.config = config

        # CNN1 Branch: Binary features
        self.cnn1 = CNNBranch(
            input_features=config.cnn1_input_features,
            num_channels=config.cnn1_num_channels,
            kernel_size=config.cnn1_kernel_size,
            num_layers=config.cnn1_num_layers
        )

        # CNN2 Branch: Technical + OHLCV features
        self.cnn2 = CNNBranch(
            input_features=config.cnn2_input_features,
            num_channels=config.cnn2_num_channels,
            kernel_size=config.cnn2_kernel_size,
            num_layers=config.cnn2_num_layers
        )

        # Combined feature size after concatenation
        combined_features = config.cnn1_num_channels + config.cnn2_num_channels

        # Fusion layer (between CNN concat and LSTM)
        # Transforms concatenated CNN outputs before feeding to LSTM
        if config.fusion_hidden_size > 0:
            fusion_output_size = int(config.fusion_hidden_size)
            self.fusion = nn.Sequential(
                nn.Linear(combined_features, fusion_output_size),
                nn.LayerNorm(fusion_output_size),
                nn.GELU(),
                nn.Dropout(config.fusion_dropout)
            )
            lstm_input_size = fusion_output_size
        else:
            self.fusion = None
            lstm_input_size = combined_features

        # LSTM encoder
        self.num_directions = 2 if config.lstm_bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=config.lstm_bidirectional
        )

        # LSTM output size
        lstm_output_size = config.lstm_hidden_size * self.num_directions

        # Classifier head
        if config.classifier_hidden_size > 0:
            # Two-layer classifier with hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_size, config.classifier_hidden_size),
                nn.LayerNorm(config.classifier_hidden_size),
                nn.GELU(),
                nn.Dropout(config.classifier_dropout),
                nn.Linear(config.classifier_hidden_size, config.num_classes)
            )
        else:
            # Single layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(config.classifier_dropout),
                nn.Linear(lstm_output_size, config.num_classes)
            )

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM and fusion layer weights."""
        # Initialize fusion layer if present
        if self.fusion is not None:
            for m in self.fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        # Initialize classifier linear layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_classifier_bias(self, class_prior: float = 0.1) -> None:
        """
        Initialize classifier bias to favor minority class predictions.

        This helps combat model collapse where the model always predicts
        the majority class (hold). By initializing the bias toward the
        minority class (trade), the model starts with a slight preference
        for predicting trade.

        Parameters
        ----------
        class_prior : float, default=0.1
            Approximate ratio of minority class (trade) samples in dataset.
            For 10:1 hold:trade imbalance, this would be ~0.1.
        """
        # Clamp class_prior to valid probability range
        class_prior = max(0.01, min(0.99, class_prior))

        # Find the final Linear layer in classifier
        for module in reversed(list(self.classifier.modules())):
            if isinstance(module, nn.Linear) and module.out_features == self.config.num_classes:
                # Set bias to log-odds: log(p/(1-p))
                # For trade (class 1): positive bias
                # For hold (class 0): negative bias
                log_prior = torch.log(torch.tensor(class_prior / (1 - class_prior)))
                module.bias.data[0] = -log_prior  # hold: negative bias
                module.bias.data[1] = log_prior   # trade: positive bias
                break

    def forward(
        self,
        binary_features: torch.Tensor,
        technical_features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        binary_features : torch.Tensor
            Binary indicator signals, shape (batch, seq_len, n_binary)
        technical_features : torch.Tensor
            Technical indicators + OHLCV, shape (batch, seq_len, n_technical)
        hidden : tuple, optional
            Initial LSTM hidden state (h_0, c_0). If None, uses zeros.

        Returns
        -------
        torch.Tensor
            Output logits, shape (batch, num_classes) for binary classification
        """
        # Process through CNN branches
        cnn1_out = self.cnn1(binary_features)  # (batch, seq_len, cnn1_channels)
        cnn2_out = self.cnn2(technical_features)  # (batch, seq_len, cnn2_channels)

        # Validate shapes match for concatenation
        assert cnn1_out.shape[:2] == cnn2_out.shape[:2], (
            f"CNN output shape mismatch: cnn1={cnn1_out.shape}, cnn2={cnn2_out.shape}"
        )

        # Concatenate CNN outputs along feature dimension
        combined = torch.cat([cnn1_out, cnn2_out], dim=2)  # (batch, seq_len, cnn1_ch + cnn2_ch)

        # Apply fusion layer if present
        if self.fusion is not None:
            combined = self.fusion(combined)  # (batch, seq_len, fusion_hidden_size)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(combined, hidden)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)

        # Extract context from final hidden state
        if self.config.lstm_bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # (batch, hidden_size)
            h_backward = h_n[-1]  # (batch, hidden_size)
            context = torch.cat([h_forward, h_backward], dim=1)
        else:
            context = h_n[-1]  # (batch, hidden_size)

        # Classification
        output = self.classifier(context)  # (batch, num_classes)

        return output

    def predict(
        self,
        binary_features: torch.Tensor,
        technical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get predicted class labels.

        Parameters
        ----------
        binary_features : torch.Tensor
            Binary indicator signals, shape (batch, seq_len, n_binary)
        technical_features : torch.Tensor
            Technical indicators + OHLCV, shape (batch, seq_len, n_technical)

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape (batch,) - 0=hold, 1=trade
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(binary_features, technical_features)
            return torch.argmax(logits, dim=-1)

    def predict_proba(
        self,
        binary_features: torch.Tensor,
        technical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get prediction probabilities.

        Parameters
        ----------
        binary_features : torch.Tensor
            Binary indicator signals, shape (batch, seq_len, n_binary)
        technical_features : torch.Tensor
            Technical indicators + OHLCV, shape (batch, seq_len, n_technical)

        Returns
        -------
        torch.Tensor
            Class probabilities, shape (batch, num_classes) - [hold_prob, trade_prob]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(binary_features, technical_features)
            return torch.softmax(logits, dim=-1)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        fusion_info = (
            f"fusion: hidden={self.config.fusion_hidden_size}, dropout={self.config.fusion_dropout}"
            if self.config.fusion_hidden_size > 0 else "fusion: disabled"
        )
        return (
            f"DualCNNLSTMPredictor(\n"
            f"  cnn1: in={self.config.cnn1_input_features}, ch={self.config.cnn1_num_channels}, "
            f"k={self.config.cnn1_kernel_size}, layers={self.config.cnn1_num_layers},\n"
            f"  cnn2: in={self.config.cnn2_input_features}, ch={self.config.cnn2_num_channels}, "
            f"k={self.config.cnn2_kernel_size}, layers={self.config.cnn2_num_layers},\n"
            f"  {fusion_info},\n"
            f"  lstm: hidden={self.config.lstm_hidden_size}, layers={self.config.lstm_num_layers}, "
            f"bidir={self.config.lstm_bidirectional},\n"
            f"  classifier: hidden={self.config.classifier_hidden_size}, "
            f"dropout={self.config.classifier_dropout},\n"
            f"  num_classes={self.config.num_classes} (hold=0, trade=1),\n"
            f"  total_params={self.get_num_parameters():,}\n"
            f")"
        )
