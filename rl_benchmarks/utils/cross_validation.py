# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for slide-level cross-validation experiments."""

from typing import Any, Dict, Tuple

import base64
import hashlib
import random
import shutil
import sys

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from .loading import load_pickle


def _aux_explore_params_recursive(
    parent_name: str, element: Any, result: Dict[str, Any]
) -> None:
    """Recursively explore parameters within a nested dictionary.

    This function recursively flattens a nested dictionary of parameters
    and their values into a flat dictionary with hierarchical keys.

    Parameters
    ----------
    parent_name: str
        The current parent key in the dictionary.
    element: Any
        The current element being processed.
    result: Dict[str, Any]
        The resulting flattened dictionary.
    """
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig):
                _aux_explore_params_recursive(f"{parent_name}.{k}", v, result)
            else:
                result[f"{parent_name}.{k}"] = v
    else:
        result[parent_name] = element


def explore_params_recursive(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explore parameters within a nested dictionary.

    This function flattens a nested dictionary of parameters and their values
    into a flat dictionary with hierarchical keys.

    Parameters
    ----------
    params: Dict[str, Any]
        The nested dictionary of parameters.

    Returns
    -------
    Dict[str, Any]
        The resulting flattened dictionary.
    """
    result = {}
    for k, v in params.items():
        _aux_explore_params_recursive(k, v, result)
    return result


def params2experiment_name(params: Dict[str, Any]) -> str:
    """Generate an experiment name based on parameters.

    This function generates a unique experiment name based on the parameters
    using a hash of their values.

    Parameters
    ----------
    params: Dict[str, Any]
        The parameters for the experiment.

    Returns
    -------
    str
        The generated experiment name.
    """

    def _make_hashable(d: Dict[str, Any]) -> Tuple:
        out = tuple(sorted(d.items()))
        return out

    def _make_hash(t: Tuple) -> str:
        hasher = hashlib.sha256()
        hasher.update(repr(t).encode())
        hash_code = base64.b64encode(hasher.digest()).decode()
        hash_code = hash_code.replace("/", "0")
        return hash_code

    result = explore_params_recursive(params)
    result_hashable = _make_hashable(result)
    experiment_name = _make_hash(result_hashable)
    return experiment_name


def resume_training(experiment_folder: str) -> None:
    """Resume training from an experiment folder.

    This function checks the status of an experiment by examining the folder.
    If the experiment was previously completed, it logs the result location
    and exits. If not completed, it cleans the folder for a fresh start.

    Parameters
    ----------
    experiment_folder: str
        The path to the experiment folder.
    """
    if experiment_folder.is_dir():
        path = experiment_folder / "status.pkl"
        if path.exists():
            status = load_pickle(path)["completed"]
            if status:
                logger.info("Experiment already done.")
                logger.info(
                    f"Experiment results available at: {experiment_folder}"
                )
                logger.info("\n")
                sys.exit()
            else:
                shutil.rmtree(experiment_folder)
                experiment_folder.mkdir(parents=True, exist_ok=True)
    else:
        experiment_folder.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set seed globally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
