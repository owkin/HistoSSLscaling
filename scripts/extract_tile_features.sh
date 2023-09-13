#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Tile-level datasets.
datasets="nct_crc camelyon17_wilds"

# Feature extractors.
feature_extractors="iBOTViTBasePANCAN"

# If `output_dir=null`, then default path is used, ie:
# conf.yaml::`data_dir`/`cohort`/features/`feature_extractor`
# conf.yaml::`data_dir`/preprocessed/tiles_classification/features/`feature_extractor`/`dataset`
# Where `data_dir` is set in `conf.yaml` file at the repository root, `dataset` is defined in
# `tile_dataset` (see `conf/extract_features` yaml configuration files),
# and `feature_extractor` belongs to the above list (see `conf/extract_features/feature_extractor/`).

# Example 1: /home/user/rl_benchmarks_data/preprocessed/tiles_classification/features/ResNet50/NCT-CRC_FULL/

# `tile_size` = "auto" means that the tile size (in pixels) in automatically
# chosen according to the feature extractor at hand. See constants::TILE_SIZES
# for details.

for dataset in $datasets
do
    for feature_extractor in $feature_extractors
    do
        echo "Feature extractor: $feature_extractor; Dataset: $dataset"
        HYDRA_FULL_ERROR=1 python ./tools/extract_features/extract_tile_features.py \
            tile_dataset=$dataset \
            feature_extractor=$feature_extractor \
            tile_size="auto" \
            batch_size=64 \
            seed=0 \
            num_workers=8 \
            device="[0,1]" \
            features_output_dir=null 
        wait
    done
done
