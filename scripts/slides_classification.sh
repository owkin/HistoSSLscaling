#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### Data loading ###
n_tiles=1000

### CV inner and outer folders ###
n_repeats_outer=1
n_splits_outer=5
n_repeats_inner=1
n_splits_inner=5

### Stratification mode ###
stratified=True
split_mode="patient_split"

### MIL model training parameters ###
# Warnings:
# - For HIPT, batch size will be automatically set to 1
# - For ResNet50 model pre-trained on ImageNet, num_epochs will be automatically set to 100
batch_size=16
num_epochs=50

# Learning rate and weight decay values for grid search
learning_rate_gs="[1.0e-3,1.0e-4]"
weight_decay_gs="[0.,1.0e-4]"

### Features and logs root directories ###
# If `features_root_dir=null`, then default path is used, ie:
# conf.yaml::`data_dir`/`cohort`/features/`feature_extractor`
# Example 1: /home/user/rl_benchmarks_data/preprocessed/slides_classification/CAMELYON16_FULL/features/ResNet50
# Example 2: /home/user/rl_benchmarks_data/preprocessed/slides_classification/TCGA/features/TCGA_COAD/ResNet50 (specific to TCGA cohorts)
# Same if `logs_root_dir=null`, then default path is used.

features_root_dir=null
logs_root_dir=null

### Device for single-GPU training ###
device="cuda:2"

slide_classification_tasks="
    camelyon16_train_tumor_prediction
    
    tcga_crc_msi_prediction
    tcga_stad_msi_prediction

    tcga_ov_hrd_prediction
    tcga_brca_hrd_prediction

    tcga_brca_histological_subtype_prediction
    tcga_brca_molecular_subtype_prediction
    tcga_nsclc_cancer_subtype_prediction
    tcga_rcc_cancer_subtype_prediction
    
    tcga_brca_os_prediction
    tcga_coad_os_prediction
    tcga_luad_os_prediction
    tcga_lusc_os_prediction
    tcga_paad_os_prediction
"
feature_extractors="
    iBOTViTBasePANCAN ResNet50 MoCoWideResNetCOAD \
    iBOTViTSmallCOAD iBOTViTBaseCOAD iBOTViTLargeCOAD
"
# If you want to add other models after taking care of feature extraction
# and duly respect the data structure detailed in the README.
# HIPT CTransPath DinoChenBRCA DinoChenPancancer 
mil_models="
    mean_pool
    chowder
    abmil
    dsmil
    hipt_mil
    trans_mil
"

for slide_classification_task in $slide_classification_tasks
do
    for feature_extractor in $feature_extractors
    do
        for mil_model in $mil_models
        do
        # Batch size should be 1 for ``'hipt_mil'`` MIL model.
        if [ $mil_model == "hipt_mil" ];then
            mil_batch_size=1
        else
            mil_batch_size=$batch_size
        fi
        # All in-domain extractors: MIL models are trained on 50 epochs,
        # Supervised out-of-domain ResNet50 extractor: MIL models are trained on 100 epochs.
        if [ $feature_extractor == "ResNet50" ];then
            mil_num_epochs=100
        else
            mil_num_epochs=$num_epochs
        fi

        HYDRA_FULL_ERROR=1 python ./tools/slide_level_tasks/cross_validation.py \
            --config-name config \
            task=$slide_classification_task \
            model=$mil_model \
            features_root_dir=$features_root_dir \
            logs_root_dir=$logs_root_dir \
            task.data.n_tiles=$n_tiles \
            task.data.feature_extractor=$feature_extractor \
            task.validation_scheme.n_repeats_outer=$n_repeats_outer \
            task.validation_scheme.n_splits_outer=$n_splits_outer \
            task.validation_scheme.n_repeats_inner=$n_repeats_inner \
            task.validation_scheme.n_splits_inner=$n_splits_inner \
            task.validation_scheme.stratified=$stratified \
            task.validation_scheme.split_mode=$split_mode \
            task.validation_scheme.grid_search_params.learning_rate=$learning_rate_gs \
            task.validation_scheme.grid_search_params.weight_decay=$weight_decay_gs \
            task.validation_scheme.trainer_cfg.batch_size=$mil_batch_size \
            task.validation_scheme.trainer_cfg.num_epochs=$mil_num_epochs \
            task.validation_scheme.trainer_cfg.device=$device
        wait
        done
    done
done
