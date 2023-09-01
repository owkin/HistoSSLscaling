"""Module covering losses for slide-level downstream tasks."""

from .bce_with_logits_loss import BCEWithLogitsLoss
from .cox_loss import CoxLoss
from .cross_entropy_loss import CrossEntropyLoss

# OS prediction.
SURVIVAL_LOSSES = CoxLoss
# Binary and multi-categorial outcome prediction, respectively.
CLASSIFICATION_LOSSES = (BCEWithLogitsLoss, CrossEntropyLoss)
