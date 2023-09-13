# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures for tests."""

from typing import Callable, Dict, List, Optional
import pytest


@pytest.fixture
def non_regression_tracking() -> Callable:
    """Non regression tracking to ensure reproducibility of cross-validation
    schemes, such that nested cross-validation."""

    def tracking(
        test_metrics: Dict[str, Dict[str, List[float]]],
        reference_metrics: Dict[str, Dict[str, List[float]]],
        digits: Optional[int] = None,
    ):
        """Assert the strict equality of 2 dictionaries of metrics values, as
        produced by cross-validation schemes. See
        ``rl_benchmarks.val_schemes.nested_cross_validation::NestedCrossValidation.run()``
        for details about the structure of ``test_metrics`` (hence ``reference_metrics``).

        Parameters
        ----------
        test_metrics: Dict[str, Dict[str, List[float]]]
            Output test metrics dictionary as derived from the validation scheme
            test running.
        reference_metrics: Dict[str, Dict[str, List[float]]]
            Expected metrics.
        digits: Optional[int] = None
            Maximum number of digits where metrics quality should be strict.
        """

        def _round_dict_values(
            obj: Dict[str, Dict[str, List[float]]]
        ) -> Dict[str, Dict[str, List[float]]]:
            """Round metrics values to ``digits`` number of digits."""
            return {
                m: {
                    rs: [round(metric, digits) for metric in metrics]
                    for (rs, metrics) in v.items()
                }
                for (m, v) in obj.items()
            }

        _test_metrics = _round_dict_values(test_metrics)
        _reference_metrics = _round_dict_values(reference_metrics)
        assert _test_metrics == _reference_metrics

    return tracking
