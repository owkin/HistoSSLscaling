# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for linear evaluation performed in
``rl_benchmarks/tools/tile_level_tasks/linear_evaluation.py``."""

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
from functools import partial
import numpy as np
import pandas as pd
import sklearn.metrics


def bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    statistic: Callable = sklearn.metrics.roc_auc_score,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Documentation inspired from
    https://github.com/scipy/scipy/blob/v1.11.2/scipy/stats/_resampling.py#L279-L660.

    Compute a two-sided bootstrap confidence interval of a statistic that takes
    `labels` and `predictions` as inputs. Bootstrap confidence interval is computed
    according to the following procedure.

    1. Resample the data: for each label and score (or predicted label) and for each of
       `n_resamples`, take a random sample of the original sample
       (with replacement) of the same size as the original sample.
    2. Compute the bootstrap distribution of the statistic: for each set of
       resamples, compute the test statistic.
    3. Determine the confidence interval: find the interval of the bootstrap
       distribution that is symmetric about the median.

    Parameters
    ----------
    labels: np.ndarray
        Ground-truth labels.
    predictions: np.ndarray
        Paired predictions (scores or predicted labels).
    statistic : Callable = sklearn.metrics.roc_auc_score
        Statistic for which the confidence interval is to be calculated,
        accuracy for instance.
    n_resamples : int = 1000
        The number of resamples performed to form the bootstrap distribution
        of the statistic.
    confidence_level : float =  0.95
        The confidence level of the confidence interval.

    Returns
    -------
    Tuple[float, float, float]
        Boostrap estimator, lower bound and high bound of confidence interval.
    """
    alpha = (1 - confidence_level) / 2
    interval = np.array([alpha, 1 - alpha])
    boostrap_values = []
    for _ in range(n_resamples):
        idx = np.random.choice(
            np.arange(len(labels)), size=len(labels), replace=True
        )
        boostrap_values.append(statistic(labels[idx], predictions[idx]))
    theta = statistic(labels, predictions)
    lower_ci, upper_ci = np.percentile(boostrap_values, interval * 100)
    interval = (2 * theta - upper_ci, 2 * theta - lower_ci)
    return theta, interval[0], interval[1]


def remove_labels(
    features: np.ndarray, labels: np.ndarray, class_index: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove elements in both ``features`` and ``labels`` with class label
    equal to ``class_index``."""
    if class_index is not None:
        idx = labels == class_index
        new_labels = labels[~idx]
        new_labels[new_labels > class_index] -= 1
        return features[~idx], new_labels
    return features, labels


def build_datasets(
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    features_root_dir: Union[str, Path],
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Create training and validation datasets for linear evaluation (tile-
    level experiments).
    Parameters
    ----------
    train_dataset: pd.DataFrame
        Training data frame as built in ``rl_benchmarks.datasets.tiles_classification``
        loading functions. It contains the following columns:
        "image_id": image ID
        "image_path": path to the tile
        "center_id": center ID (optional)
        "label": tissue class (0 to 8, NCT-CRC) or presence of tumor (0 or 1, Camelyon17-WILDS)

    val_dataset: pd.DataFrame
        Validation data frame with same specifications as above.
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        tiles_classification/features/iBOTViTBasePANCAN/NCT-CRC_FULL/
    """
    tile_ids = np.load(features_root_dir / "tile_ids.npy", allow_pickle=True)
    tile_features = np.load(features_root_dir / "tile_features.npy")
    tile_data = {
        t_id: t_feat for (t_id, t_feat) in zip(tile_ids, tile_features)
    }

    _train_features = np.array(
        [tile_data[t_id] for t_id in train_dataset.index]
    )
    _val_features = np.array([tile_data[t_id] for t_id in val_dataset.index])
    # Remove background label for NCT-CRC, other cohorts' labels are not
    # modified.
    class_index = 1 if "NCT-CRC" in str(features_root_dir) else None
    train_features, train_labels = remove_labels(
        features=_train_features,
        labels=train_dataset.label.values,
        class_index=class_index,
    )
    val_features, val_labels = remove_labels(
        features=_val_features,
        labels=val_dataset.label.values,
        class_index=class_index,
    )
    return train_features, val_features, train_labels, val_labels


def get_binary_class_metrics(
    val_labels: np.array, val_scores: np.array
) -> List[Tuple[float, float, float]]:
    """Get ROC AUC score, accuracy score and F1 score for each class of
    multi-class labels.
    Parameters
    ----------
    val_labels: np.array
        Ground-truth labels, shape (N,)
    val_scores: np.array
        Probabilities, shape (N, N_classes).

    Returns
    -------
    List[Tuple[float, float, float]]
        Each element of the list contains the metrics for a specific label.
        Length of the list is equal to the number of unique labels found in
        ``val_labels``.
    """
    val_predictions = val_scores.argmax(axis=1)
    # Binary labels.
    binary_metrics = []
    # Iterate over the set of unique labels (even if those are in {0, 1}).
    for i, label in enumerate(np.unique(val_labels)):
        val_label_class = (val_labels == label) * 1
        val_binary_predictions = (val_predictions == label) * 1
        binary_metrics.append(
            [
                sklearn.metrics.roc_auc_score(
                    val_label_class, val_scores[:, i]
                ),  # auc
                sklearn.metrics.accuracy_score(
                    val_label_class, val_binary_predictions
                ),  # acc
                sklearn.metrics.f1_score(
                    val_label_class, val_binary_predictions
                ),  # f1
            ]
        )
    return binary_metrics


def get_bootstrapped_metrics(
    val_labels: np.array,
    val_scores: np.array,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> List[Tuple[float, float, float]]:
    """Perform boostrap to get estimate and confidence interval of ROC AUC
    score, accuracy score and F1 score.
    Parameters
    ----------
    val_labels: np.array
        Ground-truth labels, shape (N,). Can be multi-class.
    val_scores: np.array
        Probabilities, shape (N, N_classes).
    n_resamples: int = 1000
        Number of bootstrap resamples (with replacement).
    confidence_level: float = 0.95
        Confidence level of confidence interval.

    Returns
    -------
    List[Tuple[float, float, float]]
        List of triples (bootstrap estimate, lower_ci, upper_ci) for each metric
        (i.e., ROC AUC score, accuracy score and F1 score, in this order).
    """
    # Bootstrapped metrics.
    # AUC.
    val_predictions = val_scores.argmax(axis=1)
    multi_class = val_scores.shape[1] > 2  # more than 2 classes
    val_scores = val_scores[:, 1] if not multi_class else val_scores
    bt_auc = bootstrap(
        val_labels,
        val_scores,
        statistic=partial(
            sklearn.metrics.roc_auc_score,
            average="macro",
            multi_class="ovr" if multi_class else "raise",
        ),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )
    # Accuracy.
    bt_acc = bootstrap(
        val_labels,
        val_predictions,
        statistic=partial(sklearn.metrics.accuracy_score),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )
    # F1-score.
    bt_f1 = bootstrap(
        val_labels,
        val_predictions,
        statistic=partial(
            sklearn.metrics.f1_score,
            average="macro" if multi_class else "binary",
        ),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )
    return bt_auc, bt_acc, bt_f1


def dict_to_dataframe(
    results_dict: Dict[
        str,
        Union[
            Dict[str, List[Tuple[float, float, float]]],
            Dict[str, Tuple[Tuple[float, float, float]]],
        ],
    ],
    metrics: List[str] = ["auc", "acc", "f1"],
    class_names: List[str] = None,
) -> pd.DataFrame:
    """Format results dictionary into a pandas Data Frame.
    Parameters
    ----------
    results_dict: Dict[str, ...]
        This dictionary has 2 keys, namely 'binary' and 'bootstrap'.
        Each subdictionary has keys corresponding to ``portion``, e.g. 0.001, 0.01, ..., 1.0
        corresponding to the fraction of training dataset used in linear evaluation.

        For each portion ``p``:
        - ``results_dict['binary'][p]``: contains the output of ``get_binary_class_metrics```
          function, i.e., a list of triplets (AUC, ACC, F1) for each unique label.
        - ``results_dict['boostrap'][p]``: contains the output of ``get_bootstrapped_metrics```
          function, i.e., a triplet (AUC, ACC, F1) with bootstrap estimates and confidence
          intervals. For multi-class problems, this bootstrap triplet is only computed for
          multi-class metrics and not per unique label (to save computation time).

    metrics: List[str] = ["auc", "acc", "f1"]
        Names of metrics (will define columns names in the output data frame).
    class_names: List[str] = None
        Names of classes, e.g. ``["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]``
        for NCT-CRC classification task.

    Returns
    -------
    pd.DataFrame
        Output data frame with all metrics (without boostrap for per-label metrics,
        with bootstrap for multi-class metrics or binary classification problems.
    """
    assert class_names is not None, "Please specify class names."
    # Binary class metrics.
    results = pd.DataFrame(results_dict["binary"]).T
    results.columns = class_names
    for class_name in class_names:
        results[[f"{class_name}_{metric}" for metric in metrics]] = results[
            class_name
        ].apply(pd.Series)
    results = results.iloc[:, len(class_names) :]
    # Bootstrapped metrics.
    bt_class_results = None
    if results_dict["bootstrap"]:
        bt_class_results = pd.DataFrame(results_dict["bootstrap"]).T
        bt_class_results.columns = metrics
        for metric in metrics:
            bt_class_results[
                [f"{metric}", f"bt_{metric}_low", f"bt_{metric}_high"]
            ] = bt_class_results[metric].apply(pd.Series)
        results = pd.concat([results, bt_class_results], axis=1)
    results = results.reset_index(drop=False, names="portion")
    return results
