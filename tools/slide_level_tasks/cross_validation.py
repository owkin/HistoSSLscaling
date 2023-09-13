# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Script to perform nested cross-validation on slide-level tasks."""

import copy
from pathlib import Path

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from rl_benchmarks.utils import (
    log_slide_dataset_info,
    params2experiment_name,
    resume_training,
    save_pickle,
    store_params,
)
from rl_benchmarks.constants import (
    LOGS_ROOT_DIR,
    PREPROCESSED_DATA_DIR,
)


@hydra.main(
    version_base=None,
    config_path="../../conf/slide_level_task/cross_validation/",
    config_name="config",
)
def main(params: DictConfig) -> None:
    """Perform nested cross-validation for a given task.
    The Hydra configuration files defining all slide-level downstream tasks
    can be found in ``'conf/slide_level_tasks/cross_validation/*.yaml'``."""
    # Get parameters.
    model_cfg = params["model"]
    task_cfg = params["task"]
    data_cfg = task_cfg["data"]
    validation_scheme_cfg = task_cfg["validation_scheme"]
    validation_scheme_cfg["trainer_cfg"]["model"] = model_cfg

    # Create experiment name.
    experiment_name = OmegaConf.to_container(copy.deepcopy(params))
    # Remove parameters from `experiment_name` that do not justify to
    # re-run the experiments, if different.
    experiment_name.pop("features_root_dir")
    experiment_name.pop("logs_root_dir")
    experiment_name["task"]["validation_scheme"]["trainer_cfg"].pop("device")
    experiment_name = params2experiment_name(experiment_name)

    # Create experiment folder and check if the experiment is already completed.
    logs_root_dir = params["logs_root_dir"]
    if logs_root_dir is None:
        logs_root_dir = LOGS_ROOT_DIR

    experiment_folder = (
        Path(logs_root_dir) / "cross_validation" / experiment_name
    )
    resume_training(experiment_folder)

    # Store experiment status (will be set to True in case of success).
    experiment_status = {"completed": False}
    path = experiment_folder / "status.pkl"
    save_pickle(path, experiment_status)

    # Create log file.
    path = experiment_folder / f"{experiment_name}.log"
    log_path_id = logger.add(path)

    # Start logging.
    logger.info("Running cross-validation script...\n")
    logger.info(
        f"Experiment name: {experiment_name}.\n"
        f"Experiment folder: {experiment_folder}\n"
    )

    # Store parameters.
    store_params(experiment_folder, params)

    # Log main task Hydra configuration.
    logger.info("---- Task configuration info ----")
    logger.info("Run configuration: \n" + OmegaConf.to_yaml(params))
    logger.info("\n")

    # Load the data to perform nested CV on.
    feature_extractor_name = data_cfg["feature_extractor"]
    features_root_dir = params["features_root_dir"]
    if features_root_dir is None:
        features_root_dir = PREPROCESSED_DATA_DIR
    features_root_dir = (
        Path(features_root_dir)
        / "slides_classification"
        / "features"
        / feature_extractor_name
    )

    # Instantiate and log dataset info.
    dataset = instantiate(
        data_cfg["train"], features_root_dir=features_root_dir
    )
    logger.info("---- Dataset info ----")
    log_slide_dataset_info(dataset)

    # Instantiate validation scheme, which is here NestedCV.
    validation_scheme = instantiate(
        validation_scheme_cfg,
        _recursive_=False,
    )
    # Run Nested CV.
    train_metrics, test_metrics = validation_scheme.run(dataset=dataset)

    # Store results as ``pickle`` objects.
    save_pickle(experiment_folder / "train_metrics.pkl", train_metrics)
    save_pickle(experiment_folder / "test_metrics.pkl", test_metrics)

    # Remove logger's sink.
    logger.remove(log_path_id)

    # Store experiment status.
    experiment_status = {"completed": True}
    path = experiment_folder / "status.pkl"
    save_pickle(path, experiment_status)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
