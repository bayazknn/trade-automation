"""
LSTM Model Module

Implements the LSTM-based model for binary classification of trading signals.

Architecture:
1. Input projection layer
2. Multi-layer LSTM encoder
3. Single FC classification head
4. Output: (batch, num_classes) logits for binary prediction (hold=0, trade=1)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for LSTM model."""
    input_size: int           # Number of input features
    hidden_size: int = 128    # LSTM hidden state size
    num_layers: int = 2       # Number of LSTM layers
    dropout: float = 0.2      # Dropout rate
    bidirectional: bool = False  # Use bidirectional LSTM
    num_classes: int = 2      # Number of output classes (hold=0, trade=1)
    input_seq_length: int = 12   # Length of input sequences


class LSTMSignalPredictor(nn.Module):
    """
    LSTM-based model for binary trading signal prediction.

    Architecture:
    - Input projection: Linear(input_size, hidden_size)
    - LSTM encoder: Multi-layer LSTM
    - Classification head: FC layers for binary prediction

    Predicts whether the next period is tradeable:
    - 0 = hold (no trade opportunity)
    - 1 = trade (trade opportunity exists)

    Input: (batch, input_seq_length, input_size) = (batch, 12, n_features)
    Output: (batch, num_classes) = (batch, 2) logits

    Attributes
    ----------
    config : ModelConfig
        Model configuration

    Examples
    --------
    >>> config = ModelConfig(input_size=50)
    >>> model = LSTMSignalPredictor(config)
    >>> x = torch.randn(32, 12, 50)  # batch=32, seq=12, features=50
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([32, 2])
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        super().__init__()
        self.config = config

        # Direction multiplier for bidirectional LSTM
        self.num_directions = 2 if config.bidirectional else 1

        # Input projection: project input features to hidden size
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # Layer normalization after projection
        self.input_norm = nn.LayerNorm(config.hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional
        )

        # Decoder input size (concatenate forward and backward if bidirectional)
        decoder_input_size = config.hidden_size * self.num_directions

        # Single classification head for binary prediction
        self.classifier = nn.Sequential(
            nn.Linear(decoder_input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal (helps with gradient flow)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero, except forget gate bias set to 1
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long-term dependencies)
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)
            elif 'weight' in name and param.dim() == 2:
                # Linear layer weights: Xavier uniform
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_seq_length, input_size)
        hidden : tuple, optional
            Initial hidden state (h_0, c_0). If None, uses zeros.

        Returns
        -------
        torch.Tensor
            Output logits, shape (batch, num_classes) for binary classification
        """
        batch_size = x.size(0)

        # Input projection: (batch, seq, input_size) -> (batch, seq, hidden_size)
        x = self.input_projection(x)
        x = self.input_norm(x)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)

        # Extract context from final hidden state
        if self.config.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2] is last layer forward, h_n[-1] is last layer backward
            h_forward = h_n[-2]  # (batch, hidden_size)
            h_backward = h_n[-1]  # (batch, hidden_size)
            context = torch.cat([h_forward, h_backward], dim=1)
        else:
            # Just use the last layer's hidden state
            context = h_n[-1]  # (batch, hidden_size)

        # Binary classification
        output = self.classifier(context)  # (batch, num_classes)

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_seq_length, input_size)

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape (batch,) - 0=hold, 1=trade
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_seq_length, input_size)

        Returns
        -------
        torch.Tensor
            Class probabilities, shape (batch, num_classes) - [hold_prob, trade_prob]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LSTMSignalPredictor(\n"
            f"  input_size={self.config.input_size},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  bidirectional={self.config.bidirectional},\n"
            f"  num_classes={self.config.num_classes} (hold=0, trade=1),\n"
            f"  total_params={self.get_num_parameters():,}\n"
            f")"
        )
