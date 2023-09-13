# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Nested cross-validation class."""

import copy
from typing import Dict, List, Optional

import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset

from .utils import (
    generate_permutations,
    split_cv,
    update_trainer_cfg,
)


class NestedCrossValidation:
    r"""Class implementing Nested cross-validation. Nested cross-validation
    involves two levels of cross-validation, an outer and inner cross-validation.
    Within the training outer folds, an inner cross-validation is performed for
    hyperparameters tuning and model selection. The best model configuration is
    chosen based on the average performance across the inner folds. This
    selected model is then evaluated on the corresponding validation outer fold,
    which was not used during model selection. The performance metrics obtained
    from each validation outer fold are averaged to estimate the model
    generalization performance. This eliminates the bias introduced by standard
    cross-validation procedure as the test data in each iteration of the outer
    cross-validation has not been used to optimize the performance of the model
    in any way, and may therefore provide a more reliable criterion for choosing
    the best model. In our study, we performed 5x5 nested cross-validation with
    no repeats (five inner and five outer splits). During nested-CV, we test
    different values of the initial learning rate and weight decay, namely
    $\{0.001,0.0001\}$ for learning rate and $\{0, 0.0001\}$ for weight decay,
    respectively. The optimal number of epochs is determined within each oute
    split through the 5-fold inner CV based on the validation metric (AUC as
    a default).


    Parameters
    ----------
    trainer_cfg: DictConfig
        Trainer configuration. Examples are available in
        ``rl_benchmarks/conf/slide_level_task/cross_validation/task/*.yaml``
        configurations files. ``trainer_cfg`` aims to instantiate the following
        trainer function: ``_target_: rl_benchmarks.trainers.TorchTrainer``.
    grid_search_params: Optional[Dict[str, List[float]]] = None
        Grid search parameters. Example:
        ``{'learning_rate': [1e-4, 1e-5], 'weight_decay': [0.0, 1e-6]}``.
        Best configuration is selected based on the inner cross-validations.
        If ``None``, no hyperparameters tuning is performed.
    n_repeats_outer: int = 1
        Number of repetitions of the outer cross-validation.
    n_splits_outer: int = 5
        Number of outer splits.
    n_repeats_inner: int = 1
        Number of repetitions of the inner cross-validations.
    n_splits_inner: int = 5
        Number of inner splits.
    stratified: bool = True
        Whether to stratify the splits.
    split_mode: str = "patient_split"
        Which mode of stratification. Other modes are ``'random_split'`` and
        ``'center_split'``. Default is ``'patient_split'`` to avoid any data
        leaking between folds.
    """

    def __init__(
        self,
        trainer_cfg: DictConfig,
        grid_search_params: Optional[Dict[str, List[float]]] = None,
        n_repeats_outer: int = 1,
        n_splits_outer: int = 5,
        n_repeats_inner: int = 1,
        n_splits_inner: int = 5,
        stratified: bool = True,
        split_mode: str = "patient_split",
    ):
        self.n_repeats_outer = n_repeats_outer
        self.n_splits_outer = n_splits_outer
        self.n_repeats_inner = n_repeats_inner
        self.n_splits_inner = n_splits_inner
        self.stratified = stratified
        self.split_mode = split_mode

        self.trainer_cfg = trainer_cfg
        self.grid_search_params = OmegaConf.create(grid_search_params)

    def run(self, dataset: Dataset) -> List[Dict[str, Dict[str, List[float]]]]:
        """Main function ot run the whole nested cross-validation.
        Parameters
        ----------
        dataset: Dataset
            Dataset to perform nested CV on.

        Returns
        -------
        ``cv_train_val_metrics``, ``cv_test_metrics``: Dict[str, Dict[str, List[float]]]
            Outer-folds metrics. Each dictionnary is of type
            ``cv_train_val_metrics['metric_name'][f'repeat_{r}_split_{s}'] = metric_values```
            where ``'metric_name'`` is either "auc" or "cindex" and values and
            ``metric_values`` is the list of the metric values for all epochs.
            The average training metrics are computed as follows on the outer folds:
            >>> for k, v in cv_train_val_metrics.items():
            >>>     logger.info(
            >>>         f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
            >>>         f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            >>>     )
            For each training outer folds, the last epoch is considered as the
            best epoch, hence ``[-1]``. Indeed the ``n_outer_folds`` optimal
            models, each derived with optimal sets of parameters and optimal
            number of epochs during corresponding inner CV, are re-trained for
            exactly ``optimal_epoch`` epochs, on the outer training folds.
            Those models are also evaluated at each epoch on the outer test fold,
            till ``optimal_epoch`` epochs. The mean of the last element of the
            metric values averages the test metrics for each outer test fold,
            hence giving the average nested-cv test metric.
        """
        # Start logging.
        logger.info("Running nested cross-validation.")

        # Automatic setting of in_features and out_features
        n_features, n_labels = (dataset.n_features, dataset.n_labels)
        # Update the main configuration with ``n_features`` and
        # ``n_labels`` for correctly instantiating the MIL
        # aggregation model (``in_features`` and ``out_features``,
        # respectively).
        update_trainer_cfg(self.trainer_cfg, n_features, n_labels)

        # Log trainer info.
        logger.info("---- Trainer info ----")
        logger.info(self.trainer_cfg)

        # Enter the outer cross-validation loop.
        cv_train_val_metrics: Dict[str, List] = {}
        cv_test_metrics: Dict[str, List] = {}
        for r_outer in range(self.n_repeats_outer):
            # Split the dataset into ``self.n_splits_outer`` outer folds using
            # the ``split_cv`` function. Splits are different for each repeat
            # depending on the ``random_state`` parameter.
            splits_outer = split_cv(
                dataset=dataset,
                n_splits=self.n_splits_outer,
                split_mode=self.split_mode,
                random_state=r_outer,
            )
            # Now iterate on the outer folds.
            for s_outer, (train_val_indices, test_indices) in enumerate(
                splits_outer
            ):
                logger.info(
                    f"Outer cross-validation: repeat_{r_outer+1}_split_{s_outer+1}"
                )
                # Split outer dataset into ``train_val_dataset`` and
                # ``test_dataset``. Inner cross-validation, gridsearching and
                # model selection will take place on the ``train_val_dataset``
                # where best model's generalization error will be evaluated
                # on the ``test_dataset``, unseen during inner CV.
                train_val_dataset = Subset(dataset, indices=train_val_indices)
                test_dataset = Subset(dataset, indices=test_indices)

                # Hyperparameter tuning (if applicable).
                if self.grid_search_params:
                    logger.info("Running hyperparameters tunning.")

                    # Create all possible config files given the dictionnary
                    # ``self.grid_search_params``.
                    permutations_dicts = generate_permutations(
                        OmegaConf.to_container(self.grid_search_params)
                    )

                    # Log hyperparameter tuning info.
                    logger.info("---- Hyperparameter tuning info ----")
                    logger.info(
                        f"Number of possible configurations: {len(permutations_dicts)}"
                    )

                    # Iterate over each possible set of parameters, ie config
                    # files.
                    list_cfgs, list_val_metrics = [], []
                    for i, sub_cfg in enumerate(permutations_dicts):
                        # Use main config file.
                        trainer_cfg = copy.deepcopy(self.trainer_cfg)
                        # Update the main configuration file with the current
                        # set of parameters we would like to perform CV on.
                        trainer_cfg.update(sub_cfg)
                        # Update this configuration with ``n_features`` and
                        # ``n_labels`` for correctly instantiating the MIL
                        # aggregation model (``in_features`` and ``out_features``,
                        # respectively).
                        update_trainer_cfg(trainer_cfg, n_features, n_labels)
                        # Log current configuration.
                        logger.info(f"Grid search #{i}")
                        logger.info(f"Current config: {sub_cfg}")

                        # Enter inner cross-validation.
                        cv_train_metrics: Dict[str, List] = {}
                        cv_val_metrics: Dict[str, List] = {}
                        for r_inner in range(self.n_repeats_inner):
                            # As done before, plit the dataset into
                            # ``self.n_splits_outer`` outer folds using the
                            # ``split_cv`` function. Splits are different for
                            # each repeat depending on the ``random_state``
                            # parameter.
                            splits_inner = split_cv(
                                dataset=train_val_dataset,
                                n_splits=self.n_splits_inner,
                                split_mode=self.split_mode,
                                random_state=r_inner,
                            )
                            for s_inner, (
                                train_indices,
                                val_indices,
                            ) in enumerate(splits_inner):
                                logger.info(
                                    f"Inner cross-validation: repeat_{r_inner+1}_split_{s_inner+1}"
                                )
                                # Split inner-cv dataset into training folds
                                # and a validation fold.
                                train_dataset = Subset(
                                    train_val_dataset, indices=train_indices
                                )
                                val_dataset = Subset(
                                    train_val_dataset, indices=val_indices
                                )

                                # Instantiate the trainer. For each fold of
                                # the inner cross-validation, the trainer is
                                # re-initialized.
                                trainer = instantiate(trainer_cfg)

                                # Perform training and retrieve end of
                                # training metrics.
                                train_metrics, val_metrics = trainer.train(
                                    train_dataset, val_dataset
                                )

                                # Store metrics.
                                for k, v in train_metrics.items():
                                    if k in cv_train_metrics:
                                        cv_train_metrics[k][
                                            f"repeat_{r_inner+1}_split_{s_inner+1}"
                                        ] = v
                                    else:
                                        cv_train_metrics[k] = {
                                            f"repeat_{r_inner+1}_split_{s_inner+1}": v
                                        }

                                for k, v in val_metrics.items():
                                    if k in cv_val_metrics:
                                        cv_val_metrics[k][
                                            f"repeat_{r_inner+1}_split_{s_inner+1}"
                                        ] = v
                                    else:
                                        cv_val_metrics[k] = {
                                            f"repeat_{r_inner+1}_split_{s_inner+1}": v
                                        }

                                # Log inner split metrics.
                                logger.info("---- Inner splits metrics ----")
                                logger.info(
                                    f"Repeat {r_inner+1}, Split {s_inner+1}"
                                )
                                logger.info("Training folds metrics:")
                                for k, v in train_metrics.items():
                                    logger.info(f"   {k}: {v[-1]:.3f}")
                                logger.info("Validation fold metrics:")
                                for k, v in val_metrics.items():
                                    logger.info(f"   {k}: {v[-1]:.3f}")

                        logger.info(
                            "---- Inner cross-validation train metrics ----"
                        )
                        for k, v in cv_train_metrics.items():
                            logger.info(
                                f"   {k}: {np.mean(list(v.values())):.3f} "
                                f"({np.std(list(v.values())):.3f})"
                            )

                        logger.info(
                            "---- Inner cross-validation validation metrics ----"
                        )
                        for k, v in cv_val_metrics.items():
                            logger.info(
                                f"   {k}: {np.mean(list(v.values())):.3f} "
                                f"({np.std(list(v.values())):.3f})"
                            )
                        # Gather metrics from the inner cross-validation at
                        # each of the training epochs, so that to select
                        # the best epoch.
                        metrics_names = cv_val_metrics.keys()
                        if "cindex" in metrics_names:
                            mean_cv_val_metrics_per_epochs = (
                                np.array(
                                    list(cv_val_metrics["cindex"].values())
                                )
                                .reshape(
                                    (
                                        self.n_splits_inner,
                                        trainer_cfg.num_epochs,
                                    )
                                )
                                .mean(axis=0)
                            )
                        elif "auc" in metrics_names:
                            mean_cv_val_metrics_per_epochs = (
                                np.array(list(cv_val_metrics["auc"].values()))
                                .reshape(
                                    (
                                        self.n_splits_inner,
                                        trainer_cfg.num_epochs,
                                    )
                                )
                                .mean(axis=0)
                            )
                        else:
                            raise ValueError(
                                "Neither `cindex` nor `auc` were found in metrics."
                            )
                        # Best epoch selection.
                        optimal_epoch = mean_cv_val_metrics_per_epochs.argmax()
                        # If best epoch is "0", then the model has been trained
                        # for 1 epoch.
                        # We report the metrics on the best epoch.
                        list_val_metrics.append(
                            mean_cv_val_metrics_per_epochs[optimal_epoch]
                        )
                        if optimal_epoch == 0:
                            optimal_epoch += 1
                        sub_cfg.update({"num_epochs": int(optimal_epoch)})
                        list_cfgs.append(sub_cfg)

                    # Determine optimal config.
                    optimal_cfg = list_cfgs[np.argmax(list_val_metrics)]
                    # Log optimal config,
                    logger.info("---- Optimal config ----")
                    logger.info(optimal_cfg)
                else:
                    logger.info("No gridsearch performed.")
                    # If no grid-search is performed.
                    optimal_cfg = {}

                # Initialize the trainer with the optimal configuration.
                trainer_cfg = copy.deepcopy(self.trainer_cfg)
                trainer_cfg.update(optimal_cfg)
                update_trainer_cfg(trainer_cfg, n_features, n_labels)
                trainer = instantiate(trainer_cfg)

                # Re train with the optimal configuration on the whole
                # inner folds, and assess performance on the outer fold.
                train_val_metrics, test_metrics = trainer.train(
                    train_val_dataset, test_dataset
                )

                # Store outer-folds metrics.
                for k, v in train_val_metrics.items():
                    if k in cv_train_val_metrics:
                        cv_train_val_metrics[k][
                            f"repeat_{r_outer+1}_split_{s_outer+1}"
                        ] = v
                    else:
                        cv_train_val_metrics[k] = {
                            f"repeat_{r_outer+1}_split_{s_outer+1}": v
                        }

                for k, v in test_metrics.items():
                    if k in cv_test_metrics:
                        cv_test_metrics[k][
                            f"repeat_{r_outer+1}_split_{s_outer+1}"
                        ] = v
                    else:
                        cv_test_metrics[k] = {
                            f"repeat_{r_outer+1}_split_{s_outer+1}": v
                        }

                # Log outer split metrics.
                logger.info("---- Outer splits metrics ----")
                logger.info(f"Repeat {r_outer+1}, Split {s_outer+1}")
                logger.info("Training folds metrics:")
                for k, v in train_val_metrics.items():
                    logger.info(f"   {k}: {v[-1]:.3f}")
                logger.info("Test fold metrics:")
                for k, v in test_metrics.items():
                    logger.info(f"   {k}: {v[-1]:.3f}")

        # Log outer cross-validation metrics.
        logger.info("")
        logger.success("---- FINAL RESULTS ----")
        logger.success("---- Outer cross-validation train metrics ----")
        for k, v in cv_train_val_metrics.items():
            # The last epoch is the best epoch (as training ended at the
            # best epoch identified during inner CV).
            logger.success(
                f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
                f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            )
        logger.success("---- Outer cross-validation test metrics ----")
        for k, v in cv_test_metrics.items():
            logger.success(
                f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
                f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            )

        return cv_train_val_metrics, cv_test_metrics
