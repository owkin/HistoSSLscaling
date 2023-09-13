#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### Features and logs root directories ###
# If `features_root_dir=null`, then default path is used, ie:
# conf.yaml::`data_dir`/preprocessed/tiles_classification/features/`feature_extractor`/`cohort`/
# Example 1: /home/user/rl_benchmarks_data/preprocessed/tiles_classification/features/ResNet50/CAMELYON17-WILDS_FULL/
features_root_dir=null
logs_root_dir=null

# Portions of the training set for linear evaluation; for each portion of the
# training dataset, `n_ensembling` models are trained with different
# initialization. Final output probabilities are the average across those models.
portions="[0.001,0.005,0.01,0.05,0.1,0.5,1.0]"
n_ensembling=30

# Bootstrap parameters for multi-class metrics
n_resamples=1000
seed=0
confidence_level=0.95

# Number of workers to train the SGD classifier
num_workers=16

tile_datasets="camelyon17_wilds nct_crc"
feature_extractors="
    ResNet50 MoCoWideResNetCOAD \
    iBOTViTSmallCOAD iBOTViTBaseCOAD iBOTViTBasePANCAN iBOTViTLargeCOAD
"
# If you want to add other models after taking care of feature extraction
# and duly respect the data structure detailed in the README.
# HIPT CTransPath DinoChenBRCA DinoChenPancancer 

for tile_dataset in $tile_datasets
do
    for feature_extractor in $feature_extractors
    do
        HYDRA_FULL_ERROR=1 python ./tools/tile_level_tasks/linear_evaluation.py \
            --config-name config \
            feature_extractor=$feature_extractor \
            tile_dataset=$tile_dataset \
            portions=$portions \
            features_root_dir=$features_root_dir \
            logs_root_dir=$logs_root_dir \
            portions=$portions \
            n_ensembling=$n_ensembling \
            n_resamples=$n_resamples \
            seed=$seed \
            confidence_level=$confidence_level \
            num_workers=$num_workers      
        wait
    done
done
