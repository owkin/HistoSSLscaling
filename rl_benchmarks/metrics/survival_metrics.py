# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Harell C-index metric for survival prediction."""

import numpy as np
from lifelines.utils import concordance_index


def compute_cindex(labels: np.array, logits: np.array):
    """Harell C-index computation.
    Parameters
    ----------
    labels: np.array
        Risk prediction from the model (higher score means higher risk and lower survival)
    logits: np.array
        Labels of the event occurences. Negative values are the censored values.
    """
    times, events = np.abs(labels), 1 * (labels > 0)
    try:
        return concordance_index(times, -logits, events)
    except AssertionError:
        return 0.5
