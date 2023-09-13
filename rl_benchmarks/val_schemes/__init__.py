# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module dedicated to cross-validation schemes.
Here, only nested cross-validation is needed."""

from .nested_cross_validation import NestedCrossValidation
from .utils import split_cv, generate_permutations, update_trainer_cfg
