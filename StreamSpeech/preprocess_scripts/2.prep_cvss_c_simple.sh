#!/bin/bash

lang=$1
CVSS_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
COVOST2_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/covost2
ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech
PRETRAIN_ROOT=$ROOT/pretrained_models

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/prep_cvss_c_simple.py \
    --covost-data-root $COVOST2_ROOT/ --cvss-data-root $CVSS_ROOT/ \
    --output-root $CVSS_ROOT/$lang-en \
    --src-lang $lang \
    --unit-type km1000 --reduce-unit