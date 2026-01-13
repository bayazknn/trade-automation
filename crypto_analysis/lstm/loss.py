"""
Custom Loss Function Module

Implements weighted CrossEntropyLoss for binary classification (hold=0, trade=1).

Supports:
- Class weights for handling imbalanced data
- Label smoothing for regularization
- Focal loss for hard example mining
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class BinarySignalLoss(nn.Module):
    """
    Weighted CrossEntropyLoss for binary signal classification.

    Handles class imbalance between hold (majority) and trade (minority)
    classes through class weighting.

    Parameters
    ----------
    class_weights : torch.Tensor, optional
        Weights for [hold, trade] classes. If None, uses uniform weights.
    label_smoothing : float, default=0.0
        Label smoothing factor (0.0 = no smoothing)

    Examples
    --------
    >>> loss_fn = BinarySignalLoss(class_weights=torch.tensor([1.0, 5.0]))
    >>> logits = torch.randn(32, 2)  # batch=32, classes=2
    >>> targets = torch.randint(0, 2, (32,))  # Binary targets
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize loss function.

        Parameters
        ----------
        class_weights : torch.Tensor, optional
            Per-class weights for [hold, trade]. Default: uniform weights.
        label_smoothing : float, default=0.0
            Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.num_classes = 2

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(2))

        # CrossEntropyLoss with class weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits, shape (batch, 2)
        targets : torch.Tensor
            Target labels, shape (batch,) with values 0 (hold) or 1 (trade)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        return self.ce_loss(logits, targets)

    def get_weight_summary(self) -> Dict[str, float]:
        """
        Get summary of configured weights.

        Returns
        -------
        dict
            Weight configuration
        """
        return {
            'hold_weight': float(self.class_weights[0]),
            'trade_weight': float(self.class_weights[1]),
            'label_smoothing': self.label_smoothing
        }

    def __repr__(self) -> str:
        return (
            f"BinarySignalLoss("
            f"hold_weight={self.class_weights[0]:.2f}, "
            f"trade_weight={self.class_weights[1]:.2f}, "
            f"label_smoothing={self.label_smoothing})"
        )


class FocalBinaryLoss(nn.Module):
    """
    Focal Loss for binary signal classification.

    Focal Loss modifies CrossEntropyLoss to down-weight easy examples and
    focus training on hard misclassified examples:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This is particularly effective for:
    - Extreme class imbalance (like hold vs trade)
    - Cases where the model confidently predicts the majority class

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Higher gamma = more focus on hard examples.
        gamma=0 is equivalent to CrossEntropyLoss.
    class_weights : torch.Tensor, optional
        Per-class weights for [hold, trade]
    label_smoothing : float, default=0.0
        Label smoothing factor

    Examples
    --------
    >>> loss_fn = FocalBinaryLoss(gamma=2.0, class_weights=torch.tensor([1., 5.]))
    >>> logits = torch.randn(32, 2)
    >>> targets = torch.randint(0, 2, (32,))
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize focal loss.

        Parameters
        ----------
        gamma : float, default=2.0
            Focal loss focusing parameter
        class_weights : torch.Tensor, optional
            Per-class weights for [hold, trade]
        label_smoothing : float, default=0.0
            Label smoothing factor
        """
        super().__init__()

        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = 2

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(2))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (batch, 2)
        targets : torch.Tensor
            Shape (batch,) with values 0 or 1

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch, 2)

        # Get probability of true class
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (batch,)

        # Apply label smoothing to confidence
        if self.label_smoothing > 0:
            p_t = p_t * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy loss per element
        log_probs = torch.log_softmax(logits, dim=-1)
        ce_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Apply class weights
        class_weights_expanded = self.class_weights[targets]

        # Combine focal weight and class weight
        weighted_loss = focal_weight * class_weights_expanded * ce_loss

        return weighted_loss.mean()

    def get_weight_summary(self) -> Dict[str, float]:
        """Get summary of configured weights."""
        return {
            'gamma': self.gamma,
            'hold_weight': float(self.class_weights[0]),
            'trade_weight': float(self.class_weights[1]),
            'label_smoothing': self.label_smoothing
        }

    def __repr__(self) -> str:
        return (
            f"FocalBinaryLoss("
            f"gamma={self.gamma}, "
            f"hold_weight={self.class_weights[0]:.2f}, "
            f"trade_weight={self.class_weights[1]:.2f})"
        )


# Aliases for backwards compatibility
WeightedSignalLoss = BinarySignalLoss
FocalWeightedLoss = FocalBinaryLoss
