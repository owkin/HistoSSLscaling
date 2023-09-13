# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions on serializing configurations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from ..utils import save_pickle


def store_params(
    experiment_folder: Union[str, Path], params: DictConfig
) -> None:
    """Store models parameters into a pickle object.
    Parameters
    ----------
    experiment_folder: Union[str, Path]
        Experiment folder where to save parameters.
    params: DictConfig
        Set of Hydra configuration parameters to serialize.
    """
    path = Path(experiment_folder) / "params.pkl"
    save_pickle(path, params)


def get_slide_config(
    params: DictConfig,
    output_dir: Union[str, Path],
    slide_paths: List[Union[str, Path]],
    coords_paths: List[Union[str, Path]],
    hydra_yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configurations for slide-level features extraction.
    Parameters
    ----------
    params: DictConfig
        Set of Hydra configuration parameters.
    output_dir: Union[str, Path]
        Path to output directory where other preprocessed data will be stored.
        This parameter will be stored in the output serialized dictionary.
    slide_paths: List[Union[str, Path]]
        Slides paths to reproduce the features extraction.
    coords_paths: List[Union[str, Path]]
        Coordinates paths to reproduce the features extraction.
    hydra_yaml_cfg: Optional[Dict[str, Any]] = None
        Source Hydra configuration as a dictionary.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary summarizing the features extraction process.
    """
    output_cfg = {
        "slides": [str(sp) for sp in slide_paths],
        "slide_dataset": output_dir.parts[-1],
        "coords": [str(smp) for smp in coords_paths],
        "export_dir": str(output_dir),
        "extractor": output_dir.parts[-2],
        "n_tiles": params["n_tiles"],
        "tile_size": params["tile_size"],
        "random_sampling": params["random_sampling"],
        "num_workers": params["num_workers"],
        "batch_size": params["batch_size"],
        "seed": params["seed"],
        "hydra_config": hydra_yaml_cfg,
    }
    return output_cfg


def get_tile_config(
    params: DictConfig,
    output_dir: Union[str, Path],
    tile_paths: List[Union[str, Path]],
    tile_ids: List[Union[str, int]],
    hydra_yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configurations for tile-level features extraction.
    Parameters
    ----------
    params: DictConfig
        Set of Hydra configuration parameters.
    output_dir: Union[str, Path]
        Path to output directory where other preprocessed data will be stored.
        This parameter will be stored in the output serialized dictionary.
    tile_paths: List[Union[str, Path]]
        Images paths to reproduce the features extraction.
    tiles_ids: List[int]
        Images IDs aligned with ``tile_paths``.
    hydra_yaml_cfg: Optional[Dict[str, Any]] = None
        Source Hydra configuration as a dictionary.
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary summarizing the features extraction process.
    """
    output_cfg = {
        "tile_paths": [str(sp) for sp in tile_paths],
        "tile_ids": tile_ids,
        "export_dir": str(output_dir),
        "extractor": output_dir.parts[-2],
        "num_workers": params["num_workers"],
        "tile_size": params["tile_size"],
        "batch_size": params["batch_size"],
        "seed": params["seed"],
        "hydra_config": hydra_yaml_cfg,
    }
    return output_cfg
