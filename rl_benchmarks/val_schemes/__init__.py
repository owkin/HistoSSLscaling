"""Module dedicated to cross-validation schemes.
Here, only nested cross-validation is needed."""

from .nested_cross_validation import NestedCrossValidation
from .utils import split_cv, generate_permutations, update_trainer_cfg
