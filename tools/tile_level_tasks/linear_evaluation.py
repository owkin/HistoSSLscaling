# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear evaluation for tile-level classification tasks."""


from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import sklearn.linear_model
from loguru import logger
from tqdm import tqdm
from rl_benchmarks.datasets import load_dataset
from rl_benchmarks.utils import (
    dict_to_dataframe,
    get_binary_class_metrics,
    get_bootstrapped_metrics,
    build_datasets,
    set_seed,
)
from rl_benchmarks.constants import PREPROCESSED_DATA_DIR, LOGS_ROOT_DIR


@hydra.main(
    version_base=None,
    config_path="../../conf/tile_level_task/linear_evaluation/",
    config_name="config",
)
def main(params: DictConfig) -> None:
    """Perform linear evaluation on a given tile dataset.
    The Hydra configuration file can be found in
    ``'conf/tile_level_task/linear_evaluation/config.yaml'``."""

    # Set seed globally.
    set_seed(params["seed"])

    # Get parameters of interest.
    feature_extractor_name = params["feature_extractor"]
    cohort = params["tile_dataset"]

    # Map cohort to dataset names and class names.
    if cohort == "nct_crc":
        tile_dataset = "NCT-CRC_FULL"
        class_names = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    elif cohort == "camelyon17_wilds":
        tile_dataset = "CAMELYON17-WILDS_FULL"
        class_names = ["Normal", "Tumor"]
    else:
        raise ValueError(
            f"{cohort} cohort is not associated with any configuration."
            f"Please refer to .yaml configurations in conf/extract_features/tile_dataset/"
            "for available datasets."
        )
    n_classes = len(class_names)
    logger.info(
        f"Loading {tile_dataset} dataset with {feature_extractor_name} features "
        f"({n_classes} classes)."
    )
    portions = params["portions"]

    # Create experiment folder (the experiment will be overriden).
    logs_root_dir = params["logs_root_dir"]
    if logs_root_dir is None:
        logs_root_dir = LOGS_ROOT_DIR

    experiment_name = f"{cohort}_{feature_extractor_name}"
    experiment_folder = (
        Path(LOGS_ROOT_DIR) / "linear_evaluation" / experiment_name
    )
    experiment_folder.mkdir(exist_ok=True, parents=True)

    # Get features root directory.
    features_root_dir = params["features_root_dir"]
    if features_root_dir is None:
        features_root_dir = PREPROCESSED_DATA_DIR
    features_root_dir = (
        Path(features_root_dir)
        / "tiles_classification"
        / "features"
        / feature_extractor_name
        / tile_dataset
    )

    # Build datasets.
    args = {
        "features_root_dir": features_root_dir,
        "label": None,
        "tile_size": params["tile_size"],
        "load_slide": False,  # no need to load slides
    }
    train_dataset = load_dataset(f"{tile_dataset.split('_')[0]}_TRAIN", **args)
    val_dataset = load_dataset(f"{tile_dataset.split('_')[0]}_VALID", **args)
    train_features, val_features, train_labels, val_labels = build_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        features_root_dir=features_root_dir,
    )
    # Gather metrics.
    binary_metrics = {}
    bootstrapped_metrics = {}
    logger.info(
        f"Iterating on different portions of the training dataset: {portions}."
    )
    for portion in tqdm(portions):
        idx = np.random.choice(
            np.arange(len(train_labels)),
            size=int(portion * len(train_labels)),
            replace=False,
        )
        _train_features = train_features[idx]
        _train_labels = train_labels[idx]
        scaler = sklearn.preprocessing.StandardScaler().fit(_train_features)
        _train_features = scaler.transform(_train_features)
        _val_features = scaler.transform(val_features)

        logger.info(
            f"Training portion: {portion*100}%, "
            f"N_train = {_train_features.shape[0]}, "
            f"N_val = {val_features.shape[0]}."
        )

        # Get predictions.
        _val_scores = []
        for seed in range(params["n_ensembling"]):
            sgdc = sklearn.linear_model.SGDClassifier(
                loss="log_loss",
                penalty="l2",
                learning_rate="adaptive",
                eta0=1e-4,
                n_jobs=params["num_workers"],
                early_stopping=False,
                random_state=seed,
            ).fit(_train_features, _train_labels)
            _val_scores.append(sgdc.predict_proba(_val_features))
        val_scores = np.mean(_val_scores, axis=0)

        # Compute metrics (binary classes and multi-class if required).
        binary_metrics[portion] = get_binary_class_metrics(
            val_labels=val_labels,
            val_scores=val_scores,
        )

        # Get bootstrapped metrics (for multi-class classification, bootstraps
        # are only performed for multi-class labels).
        bootstrapped_metrics[portion] = get_bootstrapped_metrics(
            val_labels=val_labels,
            val_scores=val_scores,
            n_resamples=params["n_resamples"],
            confidence_level=params["confidence_level"],
        )

    # Format the results.
    results_dict = {
        "binary": binary_metrics,
        "bootstrap": bootstrapped_metrics,
    }
    results = dict_to_dataframe(
        results_dict, metrics=["auc", "acc", "f1"], class_names=class_names
    )

    # Save the results into a ``'.csv'`` file.
    output_results_dir = f"{experiment_folder}.csv"
    results.to_csv(output_results_dir, index=None)
    logger.success(f"Results saved at {output_results_dir}.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
