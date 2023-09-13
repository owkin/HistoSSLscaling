# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test nested cross-validation scheme."""

from typing import Dict, List, Optional
import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rl_benchmarks.constants import PREPROCESSED_DATA_DIR
from rl_benchmarks.utils import load_yaml, set_seed


def _test_val_scheme_strategy(
    test_params: DictConfig,
    non_regression_tracking: pytest.fixture,
    reference_metrics: Dict[str, Dict[str, List[float]]],
    digits: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    test_params: DictConfig
        Task configuration, as defined in
        ``./conf/slide_level_task/cross_validation/test_{val_scheme}.yaml``,
        with ``val_scheme`` being ``'ncv'`` (only scheme available).
    non_regression_tracking: pytest.fixture
        Callable function comparing dictionaries with metrics values.
    reference_metrics: Dict[str, Dict[str, List[float]]]
        Output dictionary values with metrics values. See
        ``rl_benchmarks.val_schemes.nested_cross_validation::NestedCrossValidation.run()`` for details about the structure of ``reference_metrics``.
    digits: Optional[int] = None
        Maximum number of digits where metrics quality should be strict.
    """
    set_seed(seed=42)
    task_cfg = OmegaConf.create(test_params)
    data_cfg = task_cfg["data"]  # pylint: disable=unsubscriptable-object

    # Load data.
    features_root_dir = (
        PREPROCESSED_DATA_DIR
        / "slides_classification"
        / "features"
        / "iBOTViTBasePANCAN"
    )
    # Instantiate the ``SlideClassificationDataset`` dataset.
    dataset = instantiate(
        data_cfg["train"],
        features_root_dir=features_root_dir,
    )
    # Instantiate the nested cross-validation scheme.
    validation_scheme = instantiate(
        task_cfg["validation_scheme"],
        _recursive_=False,  # pylint: disable=unsubscriptable-object
    )
    # Run nested cross-validation and retrieve metrics.
    _, test_metrics = validation_scheme.run(dataset=dataset)

    # Compare to test (reference) metrics.
    non_regression_tracking(test_metrics, reference_metrics, digits=digits)


@pytest.mark.parametrize(
    "val_scheme",
    ["ncv"],
)
def test_val_schemes(
    val_scheme: str,
    non_regression_tracking: pytest.fixture,
) -> None:
    """Run a cross-validation scheme and check that, for the same given global
    seed, the results are identical up to the 4th digits to those expected.
    Parameters
    ----------
    val_scheme: str
        Cross-validation scheme. Here only ``'ncv'`` is available.
    non_regression_tracking: pytest.fixture
        Callable function comparing dictionaries with metrics values.
    """
    params = load_yaml(
        f"./conf/slide_level_task/cross_validation/test_{val_scheme}.yaml"
    )
    params = OmegaConf.create(params)

    if val_scheme == "ncv":  # nested cross-validation
        reference_metrics = {
            "cindex": {
                "repeat_1_split_1": [0.6160135760567726],
                "repeat_1_split_2": [
                    0.5260926288323549,
                    0.5622961513372472,
                    0.5817025440313112,
                    0.5831702544031311,
                ],
            }
        }

    _test_val_scheme_strategy(
        params, non_regression_tracking, reference_metrics, digits=4
    )
