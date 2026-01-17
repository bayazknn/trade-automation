"""
Dual-CNN GRU Model Module

Implements the Dual-CNN GRU architecture for binary classification of trading signals.

Architecture:
1. CNN1 Branch: Processes binary indicator signals (pure 0/1 entry/exit signals)
2. CNN2 Branch: Processes technical indicators + OHLCV (scaled continuous values)
3. Concatenation: Combines outputs from both CNN branches
4. GRU: Captures temporal dependencies from combined features
5. Classifier: Binary prediction (hold=0, trade=1)

Key Design Decisions:
- Mish activations instead of ReLU (smooth, self-regularizing)
- GRU instead of LSTM (simpler, faster, similar performance)
- BatchNorm1d in CNN branches, LayerNorm in classifier
- Light dropout only before classifier, none in CNN branches
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualModelConfig:
    """Configuration for Dual-CNN GRU model."""

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

    # Fusion Layer (between CNN concat and GRU)
    fusion_hidden_size: int = 128  # Output size of fusion layer (0 = no fusion layer)
    fusion_dropout: float = 0.1  # Dropout after fusion linear layer

    # GRU
    gru_hidden_size: int = 128  # GRU hidden state size
    gru_num_layers: int = 2  # Number of GRU layers
    gru_dropout: float = 0.1  # Dropout between GRU layers
    gru_bidirectional: bool = False  # Use bidirectional GRU

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

    Uses Conv1d layers with BatchNorm and Mish activations.
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
        layers.append(nn.Mish())

        # Additional conv layers: num_channels -> num_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(num_channels, num_channels, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(num_channels))
            layers.append(nn.Mish())

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


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connection.

    Uses dilated causal convolutions to capture temporal dependencies
    with an exact receptive field controlled by dilation factor.

    Features:
    - Causal padding: no future information leakage
    - Residual connection for gradient flow
    - BatchNorm + Mish activation
    - Dropout for regularization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        """
        Initialize TCN block.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size for convolutions
        dilation : int
            Dilation factor for dilated convolutions
        dropout : float
            Dropout rate
        """
        super().__init__()
        # Causal padding: pad only on the left side
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel mismatch, else identity
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

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
        Forward pass with causal convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, channels, seq_len)

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, out_channels, seq_len)
        """
        # First conv + BN + activation
        out = self.conv1(x)
        # Trim future timesteps for causal convolution
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = F.mish(self.bn1(out))
        out = self.dropout(out)

        # Second conv + BN
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)

        # Residual connection (trim to match output length if needed)
        res = self.residual(x)
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]

        return F.mish(out + res)


class TCNBranch(nn.Module):
    """
    TCN branch for temporal feature extraction.

    Stacks multiple TCN blocks with exponentially increasing dilations
    to achieve a large receptive field with few parameters.

    Dilations: [1, 2, 4, 8, ...] for num_layers blocks
    Receptive field: (kernel_size - 1) * sum(dilations) + 1
    """

    def __init__(
        self,
        input_features: int,
        num_channels: int,
        kernel_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize TCN branch.

        Parameters
        ----------
        input_features : int
            Number of input features (channels)
        num_channels : int
            Number of output channels per TCN block
        kernel_size : int
            Kernel size for convolutions
        num_layers : int
            Number of TCN blocks (dilations = [2^0, 2^1, ..., 2^(num_layers-1)])
        dropout : float
            Dropout rate in each TCN block
        """
        super().__init__()
        dilations = [2**i for i in range(num_layers)]  # [1, 2, 4, 8, ...]

        layers = []
        for i, dilation in enumerate(dilations):
            in_ch = input_features if i == 0 else num_channels
            layers.append(
                TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout)
            )

        self.network = nn.ModuleList(layers)
        self.output_channels = num_channels

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, features)
        return_sequence : bool
            If True, return full sequence. If False, return only last timestep.

        Returns
        -------
        torch.Tensor
            If return_sequence=True: (batch, seq_len, channels)
            If return_sequence=False: (batch, channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        for layer in self.network:
            x = layer(x)

        if return_sequence:
            # Return full sequence: (batch, seq_len, channels)
            return x.transpose(1, 2)
        else:
            # Return last timestep: (batch, channels)
            return x[:, :, -1]


@dataclass
class DualTCNConfig:
    """Configuration for Dual-TCN-LSTM model."""

    # Required fields (no defaults) - must come first
    cnn1_input_features: int  # Number of binary input features
    cnn2_input_features: int  # Number of technical + OHLCV input features

    # TCN architecture
    tcn_num_channels: int = 32  # Output channels per TCN block
    tcn_kernel_size: int = 3  # Kernel size for TCN convolutions
    tcn_num_layers: int = 4  # Number of TCN blocks (dilations: 1,2,4,8)
    tcn_dropout: float = 0.2  # Dropout in TCN blocks

    # LSTM layer (between TCN and classifier)
    lstm_hidden_size: int = 64  # LSTM hidden state size
    lstm_num_layers: int = 1  # Number of LSTM layers
    lstm_dropout: float = 0.1  # Dropout between LSTM layers (if num_layers > 1)

    # Classifier
    classifier_hidden_size: int = 32  # Hidden layer size (0 = direct projection)
    classifier_dropout: float = 0.2  # Dropout before final layer

    # Output
    num_classes: int = 2  # Binary: hold=0, trade=1

    # Sequence
    input_seq_length: int = 16  # Length of input sequences


class DualTCNPredictor(nn.Module):
    """
    Dual-TCN-LSTM model for binary trading signal prediction.

    Architecture:
    - Two parallel TCN branches for binary and technical features
    - Element-wise sum fusion of branch outputs (full sequence)
    - LSTM layer for temporal aggregation
    - Classifier head for binary prediction

    Key advantages:
    - TCN: Parallel convolutions, exact receptive field via dilations
    - LSTM: Captures long-range temporal dependencies from TCN features
    - Sum fusion reduces feature dimensionality

    Input:
    - binary_features: (batch, seq_len, n_binary) - pure 0/1 signals
    - technical_features: (batch, seq_len, n_technical) - scaled indicators + OHLCV

    Output:
    - logits: (batch, num_classes) = (batch, 2) for hold/trade prediction

    Examples
    --------
    >>> config = DualTCNConfig(
    ...     cnn1_input_features=114,
    ...     cnn2_input_features=81
    ... )
    >>> model = DualTCNPredictor(config)
    >>> binary = torch.randn(32, 16, 114)
    >>> technical = torch.randn(32, 16, 81)
    >>> output = model(binary, technical)
    >>> print(output.shape)
    torch.Size([32, 2])
    """

    def __init__(self, config: DualTCNConfig):
        """
        Initialize the model.

        Parameters
        ----------
        config : DualTCNConfig
            Model configuration
        """
        super().__init__()
        self.config = config

        # TCN1 Branch: Binary features
        self.tcn1 = TCNBranch(
            input_features=config.cnn1_input_features,
            num_channels=config.tcn_num_channels,
            kernel_size=config.tcn_kernel_size,
            num_layers=config.tcn_num_layers,
            dropout=config.tcn_dropout
        )

        # TCN2 Branch: Technical + OHLCV features
        self.tcn2 = TCNBranch(
            input_features=config.cnn2_input_features,
            num_channels=config.tcn_num_channels,
            kernel_size=config.tcn_kernel_size,
            num_layers=config.tcn_num_layers,
            dropout=config.tcn_dropout
        )

        # LSTM layer for temporal aggregation of TCN features
        self.lstm = nn.LSTM(
            input_size=config.tcn_num_channels,  # TCN output channels (after sum)
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Classifier input is LSTM hidden size
        classifier_input_size = config.lstm_hidden_size

        # Classifier head
        if config.classifier_hidden_size > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, config.classifier_hidden_size),
                nn.LayerNorm(config.classifier_hidden_size),
                nn.Mish(),
                nn.Dropout(config.classifier_dropout),
                nn.Linear(config.classifier_hidden_size, config.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(config.classifier_dropout),
                nn.Linear(classifier_input_size, config.num_classes)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and classifier weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_classifier_bias(self, class_prior: float = 0.1) -> None:
        """
        Initialize classifier bias to favor minority class predictions.

        Parameters
        ----------
        class_prior : float, default=0.1
            Approximate ratio of minority class (trade) samples in dataset.
        """
        class_prior = max(0.01, min(0.99, class_prior))

        for module in reversed(list(self.classifier.modules())):
            if isinstance(module, nn.Linear) and module.out_features == self.config.num_classes:
                log_prior = torch.log(torch.tensor(class_prior / (1 - class_prior)))
                module.bias.data[0] = -log_prior  # hold: negative bias
                module.bias.data[1] = log_prior   # trade: positive bias
                break

    def forward(
        self,
        binary_features: torch.Tensor,
        technical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        binary_features : torch.Tensor
            Binary indicator signals, shape (batch, seq_len, n_binary)
        technical_features : torch.Tensor
            Technical indicators + OHLCV, shape (batch, seq_len, n_technical)

        Returns
        -------
        torch.Tensor
            Output logits, shape (batch, num_classes) for binary classification
        """
        # Process through TCN branches (returns full sequence)
        tcn1_out = self.tcn1(binary_features, return_sequence=True)   # (batch, seq_len, channels)
        tcn2_out = self.tcn2(technical_features, return_sequence=True)  # (batch, seq_len, channels)

        # Element-wise sum of branch outputs
        combined = tcn1_out + tcn2_out  # (batch, seq_len, channels)

        # LSTM temporal aggregation
        lstm_out, (h_n, c_n) = self.lstm(combined)
        # h_n: (num_layers, batch, hidden_size) - use last layer's hidden state
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
        return (
            f"DualTCNPredictor(\n"
            f"  tcn1: in={self.config.cnn1_input_features}, ch={self.config.tcn_num_channels}, "
            f"k={self.config.tcn_kernel_size}, layers={self.config.tcn_num_layers},\n"
            f"  tcn2: in={self.config.cnn2_input_features}, ch={self.config.tcn_num_channels}, "
            f"k={self.config.tcn_kernel_size}, layers={self.config.tcn_num_layers},\n"
            f"  lstm: hidden={self.config.lstm_hidden_size}, layers={self.config.lstm_num_layers},\n"
            f"  classifier: hidden={self.config.classifier_hidden_size}, "
            f"dropout={self.config.classifier_dropout},\n"
            f"  num_classes={self.config.num_classes} (hold=0, trade=1),\n"
            f"  total_params={self.get_num_parameters():,}\n"
            f")"
        )


class DualCNNLSTMPredictor(nn.Module):
    """
    Dual-CNN GRU model for binary trading signal prediction.

    Architecture:
    - Two parallel CNN branches for binary and technical features
    - Feature concatenation after CNN processing
    - GRU encoder for temporal dependencies
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

        # Fusion layer (between CNN concat and GRU)
        # Transforms concatenated CNN outputs before feeding to GRU
        if config.fusion_hidden_size > 0:
            fusion_output_size = int(config.fusion_hidden_size)
            self.fusion = nn.Sequential(
                nn.Linear(combined_features, fusion_output_size),
                nn.LayerNorm(fusion_output_size),
                nn.GELU(),
                nn.Dropout(config.fusion_dropout)
            )
            gru_input_size = fusion_output_size
        else:
            self.fusion = None
            gru_input_size = combined_features

        # GRU encoder
        self.num_directions = 2 if config.gru_bidirectional else 1
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0.0,
            bidirectional=config.gru_bidirectional
        )

        # GRU output size
        gru_output_size = config.gru_hidden_size * self.num_directions

        # Classifier head
        if config.classifier_hidden_size > 0:
            # Two-layer classifier with hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(gru_output_size, config.classifier_hidden_size),
                nn.LayerNorm(config.classifier_hidden_size),
                nn.GELU(),
                nn.Dropout(config.classifier_dropout),
                nn.Linear(config.classifier_hidden_size, config.num_classes)
            )
        else:
            # Single layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(config.classifier_dropout),
                nn.Linear(gru_output_size, config.num_classes)
            )

        self._init_gru_weights()

    def _init_gru_weights(self):
        """Initialize GRU and fusion layer weights."""
        # Initialize fusion layer if present
        if self.fusion is not None:
            for m in self.fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

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
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        binary_features : torch.Tensor
            Binary indicator signals, shape (batch, seq_len, n_binary)
        technical_features : torch.Tensor
            Technical indicators + OHLCV, shape (batch, seq_len, n_technical)
        hidden : torch.Tensor, optional
            Initial GRU hidden state h_0. If None, uses zeros.

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

        # GRU encoding (only returns h_n, not c_n like LSTM)
        gru_out, h_n = self.gru(combined, hidden)
        # gru_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)

        # Extract context from final hidden state
        if self.config.gru_bidirectional:
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
            f"  gru: hidden={self.config.gru_hidden_size}, layers={self.config.gru_num_layers}, "
            f"bidir={self.config.gru_bidirectional},\n"
            f"  classifier: hidden={self.config.classifier_hidden_size}, "
            f"dropout={self.config.classifier_dropout},\n"
            f"  num_classes={self.config.num_classes} (hold=0, trade=1),\n"
            f"  total_params={self.get_num_parameters():,}\n"
            f")"
        )


def validate_model_architecture(
    model: Union[DualCNNLSTMPredictor, DualTCNPredictor],
    batch_size: int = 2,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """
    Comprehensive architecture validation for Dual-CNN GRU or Dual-TCN models.

    Performs the following checks:
    1. Parameter count (total, trainable, per-layer breakdown)
    2. Forward pass with shape tracing
    3. Gradient flow verification
    4. Output validation (NaN/Inf checks, shape validation)

    Parameters
    ----------
    model : Union[DualCNNLSTMPredictor, DualTCNPredictor]
        Model to validate (either GRU-based or TCN-based)
    batch_size : int, default=2
        Batch size for test tensors
    device : str, default='cpu'
        Device to run validation on ('cpu' or 'cuda')
    verbose : bool, default=True
        Print detailed validation report

    Returns
    -------
    tuple
        (success: bool, errors: List[str], warnings: List[str])

    Examples
    --------
    >>> config = DualModelConfig(cnn1_input_features=10, cnn2_input_features=20)
    >>> model = DualCNNLSTMPredictor(config)
    >>> success, errors, warnings = validate_model_architecture(model)

    >>> tcn_config = DualTCNConfig(cnn1_input_features=10, cnn2_input_features=20)
    >>> tcn_model = DualTCNPredictor(tcn_config)
    >>> success, errors, warnings = validate_model_architecture(tcn_model)
    """
    errors: List[str] = []
    warnings: List[str] = []
    config = model.config

    # Move model to device
    model = model.to(device)

    # Create test inputs
    seq_len = config.input_seq_length
    binary_input = torch.randn(batch_size, seq_len, config.cnn1_input_features).to(device)
    technical_input = torch.randn(batch_size, seq_len, config.cnn2_input_features).to(device)

    if verbose:
        print("=" * 70)
        print("MODEL ARCHITECTURE VALIDATION REPORT")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Parameter Count
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    if verbose:
        print("\n" + "-" * 70)
        print("1. PARAMETER COUNT")
        print("-" * 70)
        print(f"{'Total parameters:':<30} {total_params:>15,}")
        print(f"{'Trainable parameters:':<30} {trainable_params:>15,}")
        print(f"{'Non-trainable parameters:':<30} {non_trainable_params:>15,}")

        print(f"\n{'Layer':<40} {'Parameters':>12}  {'Shape'}")
        print("-" * 70)
        for name, param in model.named_parameters():
            print(f"{name:<40} {param.numel():>12,}  {list(param.shape)}")

    # Parameter sanity checks
    if total_params < 1000:
        warnings.append(f"Very few parameters ({total_params:,}) - model may be too simple")
    if total_params > 100_000_000:
        warnings.append(f"Very large model ({total_params:,} params) - check if intended")

    # -------------------------------------------------------------------------
    # 2. Forward Pass with Shape Tracing
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 70)
        print("2. FORWARD PASS - SHAPE TRACE")
        print("-" * 70)
        print(f"{'Layer':<40} {'Output Shape':<25}")
        print("-" * 70)
        print(f"{'Input: binary_features':<40} {list(binary_input.shape)}")
        print(f"{'Input: technical_features':<40} {list(technical_input.shape)}")

    # Register hooks to capture intermediate shapes
    shapes: Dict[str, torch.Size] = {}

    def hook_fn(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # GRU returns (output, hidden_state)
                shapes[name] = output[0].shape
            else:
                shapes[name] = output.shape
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if name and not any(sub in name for sub in ['conv_layers']):
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    # Forward pass
    model.eval()
    try:
        with torch.no_grad():
            output = model(binary_input, technical_input)

        if verbose:
            for name, shape in shapes.items():
                print(f"{name:<40} {list(shape)}")
            print("-" * 70)
            print(f"{'Final Output':<40} {list(output.shape)}")

    except RuntimeError as e:
        errors.append(f"Forward pass failed: {e}")
        # Remove hooks and return early
        for hook in hooks:
            hook.remove()
        if verbose:
            _print_validation_summary(errors, warnings, verbose)
        return False, errors, warnings

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # -------------------------------------------------------------------------
    # 3. Output Validation
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 70)
        print("3. OUTPUT VALIDATION")
        print("-" * 70)

    # Check output shape
    expected_shape = (batch_size, config.num_classes)
    if list(output.shape) != list(expected_shape):
        errors.append(f"Output shape {list(output.shape)} != expected {list(expected_shape)}")
        if verbose:
            print(f"  ✗ Output shape mismatch: {list(output.shape)} != {list(expected_shape)}")
    else:
        if verbose:
            print(f"  ✓ Output shape correct: {list(output.shape)}")

    # Check for NaN/Inf
    if torch.isnan(output).any():
        errors.append("Output contains NaN values")
        if verbose:
            print("  ✗ Output contains NaN values")
    else:
        if verbose:
            print("  ✓ No NaN values in output")

    if torch.isinf(output).any():
        errors.append("Output contains Inf values")
        if verbose:
            print("  ✗ Output contains Inf values")
    else:
        if verbose:
            print("  ✓ No Inf values in output")

    # Check batch dimension preserved
    if output.shape[0] != batch_size:
        errors.append(f"Batch size changed from {batch_size} to {output.shape[0]}")
        if verbose:
            print(f"  ✗ Batch size changed from {batch_size} to {output.shape[0]}")
    else:
        if verbose:
            print(f"  ✓ Batch dimension preserved: {batch_size}")

    # -------------------------------------------------------------------------
    # 4. Gradient Flow Verification
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 70)
        print("4. GRADIENT FLOW VERIFICATION")
        print("-" * 70)
        print(f"{'Layer':<40} {'Grad Norm':>12}  {'Status'}")
        print("-" * 70)

    model.train()
    binary_input_grad = torch.randn_like(binary_input, requires_grad=True)
    technical_input_grad = torch.randn_like(technical_input, requires_grad=True)

    # Forward and backward pass
    output = model(binary_input_grad, technical_input_grad)
    target = torch.randint(0, config.num_classes, (batch_size,)).to(device)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    grad_issues = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-10:
                status = "⚠ Very small"
                grad_issues.append(name)
            else:
                status = "✓ OK"
        else:
            grad_norm = 0.0
            status = "✗ No gradient!"
            grad_issues.append(name)

        if verbose:
            print(f"{name:<40} {grad_norm:>12.6f}  {status}")

    if grad_issues:
        warnings.append(f"Gradient issues in {len(grad_issues)} parameters: {grad_issues[:3]}...")

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------
    success = len(errors) == 0
    _print_validation_summary(errors, warnings, verbose)

    if verbose:
        print(f"\nModel Configuration:")
        is_tcn = isinstance(model, DualTCNPredictor)

        if is_tcn:
            # TCN model configuration
            print(f"  TCN1: in={config.cnn1_input_features}, ch={config.tcn_num_channels}, "
                  f"k={config.tcn_kernel_size}, layers={config.tcn_num_layers}")
            print(f"  TCN2: in={config.cnn2_input_features}, ch={config.tcn_num_channels}, "
                  f"k={config.tcn_kernel_size}, layers={config.tcn_num_layers}")
        else:
            # GRU model configuration
            print(f"  CNN1: in={config.cnn1_input_features}, ch={config.cnn1_num_channels}, "
                  f"k={config.cnn1_kernel_size}, layers={config.cnn1_num_layers}")
            print(f"  CNN2: in={config.cnn2_input_features}, ch={config.cnn2_num_channels}, "
                  f"k={config.cnn2_kernel_size}, layers={config.cnn2_num_layers}")
            if config.fusion_hidden_size > 0:
                print(f"  Fusion: hidden={config.fusion_hidden_size}, dropout={config.fusion_dropout}")
            else:
                print("  Fusion: disabled")
            print(f"  GRU: hidden={config.gru_hidden_size}, layers={config.gru_num_layers}, "
                  f"bidir={config.gru_bidirectional}")

        print(f"  Classifier: hidden={config.classifier_hidden_size}, "
              f"dropout={config.classifier_dropout}")
        print(f"  Input: batch={batch_size}, seq_len={seq_len}, "
              f"binary={config.cnn1_input_features}, technical={config.cnn2_input_features}")
        print(f"  Output: ({batch_size}, {config.num_classes})")

    return success, errors, warnings


def _print_validation_summary(
    errors: List[str],
    warnings: List[str],
    verbose: bool
) -> None:
    """Print validation summary with errors and warnings."""
    if not verbose:
        return

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"   • {e}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   • {w}")

    if not errors and not warnings:
        print("\n✅ All validation checks passed!")
    elif not errors:
        print("\n✓ No critical errors found (warnings only)")
