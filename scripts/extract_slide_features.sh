#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Slide-level datasets.
datasets="
    camelyon16_full \
    tcga_coad tcga_kich tcga_kirc tcga_kirp tcga_luad \
    tcga_lusc tcga_ov tcga_paad tcga_read tcga_stad 
"

# Feature extractors.
feature_extractors="iBOTViTBasePANCAN"

# Example with (at most) 1000 tiles per slide maximum and random sampling,
# and 2 GPUs (id 0 and 1). If `features_output_dir=null`, then default path is
# used, ie:
# conf.yaml::`data_dir`/preprocessed/slides_classification/features/`feature_extractor`/`dataset`
# Where `data_dir` is set in `conf.yaml` file at the repository root, `dataset`
# is defined in `slide_dataset` (see `conf/extract_features/slide_dataset` yaml configuration files),
# and `feature_extractor` belongs to the above list (see `conf/extract_features/feature_extractor/`).

# Example 1: /home/user/rl_benchmarks_data/preprocessed/slides_classification/features/ResNet50/CAMELYON16_FULL/
# Example 2: /home/user/rl_benchmarks_data/preprocessed/slides_classification/features/ResNet50/TCGA/TCGA_COAD/ (specific to TCGA cohorts)

# `tile_size` = "auto" means that the tile size (in pixels) in automatically
# chosen according to the feature extractor at hand. See constants::TILE_SIZES
# for details. If `n_tiles = null`, then all tiles are extracted from the slide.
# In that case, specify `random_sampling = False` (no need for sampling).

for dataset in $datasets
do
    for feature_extractor in $feature_extractors
    do
        echo "---- Feature extractor: $feature_extractor; Dataset: $dataset ---"
        HYDRA_FULL_ERROR=1 python ./tools/extract_features/extract_slide_features.py \
            feature_extractor=$feature_extractor \
            slide_dataset=$dataset \
            n_tiles=1000 \
            tile_size="auto" \
            batch_size=64 \
            random_sampling=True \
            seed=0 \
            num_workers=16 \
            device="[0,1]" \
            features_output_dir=null 
        wait
    done
done
