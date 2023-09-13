# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Classification metrics."""

from typing import Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

from ..utils import sigmoid, softmax


def compute_binary_accuracy(labels: np.array, logits: np.array) -> float:
    """Binary accuracy.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    preds = np.where(logits > 0, 1, 0)
    return accuracy_score(labels, preds)


def compute_multiclass_accuracy(labels: np.array, logits: np.array) -> float:
    """Multi-class accuracy.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    preds = np.expand_dims(np.argmax(logits, axis=1), axis=1)
    return accuracy_score(labels, preds)


def compute_binary_auc(labels: np.array, logits: np.array) -> float:
    """Binary ROC AUC score.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    # ``logits`` are [logit_0] with logit_0 unnormalized score
    if logits.shape[1] == 1:
        preds = sigmoid(logits)
    # ``logits`` are [logit_0, logit_1] with logit_0 and logit_1
    # unnormalized scores
    else:
        preds = softmax(logits)[:, 1]
    try:
        return roc_auc_score(labels, preds)
    except ValueError:
        return np.nan


def compute_one_vs_all_auc(
    labels: np.array, logits: np.array, target_label: Union[int, float]
) -> float:
    """One-vs-all ROC AUC score in a multi-class classification setting.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Multi-class probabilities.
    target_label: Union[int, float]
        Target class for binary ROC AUC score computation.
    """
    one_vs_all_label = np.array(labels == target_label, int)
    try:
        score = roc_auc_score(one_vs_all_label, logits[:, int(target_label)])
    except ValueError:
        score = np.nan
    return score


def compute_mean_one_vs_all_auc(labels: np.array, logits: np.array) -> float:
    """Macro ROC AUC score defined as the average of all one-vs-all ROC AUC scores.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Multi-class probabilities.
    """
    available_labels = np.unique(labels)
    all_aucs = np.array(
        [compute_one_vs_all_auc(labels, logits, i) for i in available_labels]
    )
    return np.mean(all_aucs)
