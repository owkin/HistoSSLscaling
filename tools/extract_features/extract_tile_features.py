# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extraction script for tiles datasets."""

import json
import shutil
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from rl_benchmarks.models import ParallelExtractor
from rl_benchmarks.utils import (
    extract_from_tiles,
    get_tile_config,
    set_seed,
)
from rl_benchmarks.constants import (
    AVAILABLE_COHORTS,
    PREPROCESSED_DATA_DIR,
    TILE_SIZES,
)


@hydra.main(
    version_base=None,
    config_path="../../conf/extract_features/",
    config_name="tile_config",
)
def extract_tile_features(params: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Perform feature extraction for a given dataset of images. The Hydra configuration
    file can be found in `conf/extract_features/tile_config.yaml`.
    See `conf/extract_features/tile_dataset` and `conf/extract_features/feature_extractor`
    for the list of available tiles datasets and feature extractors, respectively.
    """
    # Set seed for features extraction (in case of random subsampling).
    set_seed()

    # Prepare output directory.
    features_output_dir = params["features_output_dir"]
    dataset_cfg = params["tile_dataset"]
    cohort = dataset_cfg["cohort"]
    if features_output_dir is None:
        if cohort not in AVAILABLE_COHORTS:
            raise ValueError(
                f"{cohort} is not found. Available cohorts can be found in"
                " ``rl_benchmarks.constants::AVAILABLE_COHORTS``."
            )
    else:
        features_output_dir = Path(features_output_dir)
    hydra_cfg = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    feature_extractor_name = hydra_cfg["feature_extractor"]
    # Features output directory.
    features_output_dir = (
        PREPROCESSED_DATA_DIR
        / "tile_classification"
        / "features"
        / feature_extractor_name
        / cohort
    )
    features_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Storage folder: {features_output_dir}.")

    # Define parameters for features extraction process.
    dataset_cfg = params["tile_dataset"]
    feature_extractor_cfg = params["feature_extractor"]
    tile_size = params["tile_size"]
    if tile_size == "auto":
        tile_size = TILE_SIZES[feature_extractor_name]
    else:
        assert (
            TILE_SIZES[feature_extractor_name] == tile_size
        ), f"Please specify a tile size (in pixels) that matches the original implementation, see constants.TILE_SIZES dictionary for details: {TILE_SIZES}"

    num_workers = params["num_workers"]
    batch_size = params["batch_size"]
    device = params["device"]

    # Get slides paths.
    get_dataset = instantiate(dataset_cfg)
    dataset = get_dataset()

    # Save output configuration.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_features_output_dir = Path(hydra_cfg["runtime"]["output_dir"])
    hydra_yaml_cfg = hydra_features_output_dir / ".hydra" / "config.yaml"

    with open(hydra_yaml_cfg, "r", encoding="utf-8") as stream:
        hydra_yaml_cfg = yaml.safe_load(stream)

    output_cfg = get_tile_config(
        params,
        features_output_dir,
        tile_paths=list(dataset.image_path.values),
        tile_ids=list(dataset.image_id.values),
        hydra_yaml_cfg=hydra_yaml_cfg,
    )
    with open(
        features_output_dir / "rlbenchmarks_extraction_params.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(output_cfg, fp)

    shutil.copyfile(
        Path(__file__).resolve(),
        features_output_dir / "rlbenchmarks_extraction_script.py",
    )

    # Get features storage paths.
    tile_features_path = features_output_dir / "tile_features.npy"
    tile_ids_path = features_output_dir / "tile_ids.npy"

    if tile_features_path.exists():
        dataset_features = np.load(tile_features_path)
        if len(dataset) == len(dataset_features):
            logger.info(
                f"Extraction already done. Features saved at: {tile_features_path}"
            )

    # Instantiate feature extractor.
    extractor = instantiate(feature_extractor_cfg)
    extractor = ParallelExtractor(
        extractor,
        gpu=device,
    )

    # Extract features.
    tile_features, tile_ids = extract_from_tiles(
        dataset_tiles=dataset,
        feature_extractor=extractor,
        tile_size=tile_size,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    # Save features and tiles ids.
    np.save(str(tile_features_path), tile_features)
    np.save(str(tile_ids_path), tile_ids)


if __name__ == "__main__":
    extract_tile_features()  # pylint: disable=no-value-for-parameter
