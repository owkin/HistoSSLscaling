# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loop into experiments directories and gather all nested cross-validation
results."""

import ast
from typing import Any, List, Union
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np


from rl_benchmarks.utils import load_pickle
from rl_benchmarks.constants import LOGS_ROOT_DIR


def get_results(experiment_dir: Path) -> Union[None, List[Any]]:
    """Retrieve the configuration parameters and results corresponding to
    a nested cross-validation experiment.
    Parameters
    ----------
    experiment_dir: Path
        Experiment's directory with results.

    Returns
    -------
    Union[None, List[Any]]
        Either ``None`` if experiment has not completed.
        Or list containing experiment's parameters and corresponding results.
    """
    # Get status of the experiment.
    status = load_pickle(experiment_dir / "status.pkl")
    if not status["completed"]:
        return None
    # Get parameters configuration as a ``pickle`` object.
    params = load_pickle(experiment_dir / "params.pkl")
    params = ast.literal_eval(str(params))
    # Parse the configuration and retrive all parameters and results.
    # MIL aggregation model.
    model = params["model"]["_target_"].split(".")[-1]
    # Validation scheme.
    validation_scheme_cfg = params["task"]["validation_scheme"]
    # Feature extractor name.
    features = params["task"]["data"]["feature_extractor"]
    # Number of tiles.
    n_tiles = params["task"]["data"]["n_tiles"]
    # Cohort.
    cohort = params["task"]["data"]["train"]["cohort"]
    # Label (if applicable).
    label = ""
    if "label" in params["task"]["data"]["train"]:
        label = params["task"]["data"]["train"]["label"]
    # No. repeat and folds.
    n_repeats_inner = validation_scheme_cfg["n_repeats_inner"]
    n_repeats_outer = validation_scheme_cfg["n_repeats_outer"]
    n_repeats = f"{n_repeats_inner}x{n_repeats_outer}"
    n_splits_inner = validation_scheme_cfg["n_splits_inner"]
    n_splits_outer = validation_scheme_cfg["n_splits_outer"]
    n_splits = f"{n_splits_inner}x{n_splits_outer}"
    # Gridsearch parameters.
    gs_params = validation_scheme_cfg["grid_search_params"]
    # Stratification parameters.
    stratified = validation_scheme_cfg["stratified"]
    split_mode = validation_scheme_cfg["split_mode"]
    # Training parameters.
    trainer_cfg = validation_scheme_cfg["trainer_cfg"]
    batch_size = trainer_cfg["batch_size"]
    num_epochs = trainer_cfg["num_epochs"]
    learning_rate = trainer_cfg["learning_rate"]
    weight_decay = trainer_cfg["weight_decay"]
    # Outer-folds metrics. We are interested in ``test_metrics`` to report
    # generalization results.
    test_metrics = load_pickle(experiment_dir / "test_metrics.pkl")
    test_metrics = {
        m: [metrics_values[-1] for (_, metrics_values) in v.items()]
        for (m, v) in test_metrics.items()
    }  # the last epoch is the best epochs fine-tuned during inner CV
    aucs = test_metrics["auc"] if "auc" in test_metrics.keys() else None
    accuracies = (
        test_metrics["accuracy"] if "accuracy" in test_metrics.keys() else None
    )
    cindexs = (
        test_metrics["cindex"] if "cindex" in test_metrics.keys() else None
    )
    # Take the average accross test outer folds.
    mean_auc = np.nanmean(aucs) if aucs is not None else None
    mean_cindex = np.nanmean(cindexs) if cindexs is not None else None
    mean_accuracy = np.nanmean(accuracies) if accuracies is not None else None
    # Take the standard deviation across test outer folders.
    std_auc = np.nanstd(aucs) if aucs is not None else None
    std_accuracy = np.nanstd(accuracies) if accuracies is not None else None
    std_cindex = np.nanstd(cindexs) if cindexs is not None else None
    # Final output.
    # fmt: off
    return [
        model, features, n_tiles, cohort, label, gs_params, n_repeats,
        n_splits, stratified, split_mode, batch_size, num_epochs,
        learning_rate, weight_decay, mean_auc, mean_accuracy, mean_cindex,
        std_auc, std_accuracy, std_cindex, str(experiment_dir),
    ]
    # fmt: on


if __name__ == "__main__":
    # Get all results from experiments directories in
    # ``'LOGS_ROOT_DIR / "cross_validation"'``.
    experiment_dirs_path = LOGS_ROOT_DIR / "cross_validation"
    experiment_dirs = list(experiment_dirs_path.iterdir())
    experiment_dirs = [
        exp_dir for exp_dir in experiment_dirs if exp_dir.is_dir()
    ]
    print(f"Number of experiments dirs: {len(experiment_dirs)}.")
    logger.info("Searching for experiments and processing results...")
    results = [
        get_results(experiment_dir) for experiment_dir in experiment_dirs
    ]
    results = [res for res in results if res is not None]
    # fmt: off
    results = pd.DataFrame(
        results,
        columns=[
            "model", "features", "n_tiles", "cohort", "label", "gs_params",
            "n_repeats", "n_splits", "stratified", "split_mode", "batch_size",
            "num_epochs", "learning_rate", "weight_decay",
            "mean_auc", "mean_accuracy", "mean_cindex", "std_auc",
            "std_accuracy", "std_cindex", "log_path",
        ],
    )
    # Save results.
    output_path = experiment_dirs_path / "nested_cross_validation_results.csv"
    results.to_csv(output_path, index=None)
    logger.success(f"Results saved at {str(output_path)}.")
    subset_cols = [
        "model", "n_tiles", "features", "label", "cohort",
        "mean_auc", "std_auc", "mean_accuracy", "std_accuracy",
        "mean_cindex", "std_cindex"
    ]
    results = results.loc[:, subset_cols]
    logger.info(f"--- Results --- \n{results}")
