lang=$1
CVSS_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/prep_cvss_c_multitask_data.py \
    --data-dir $CVSS_ROOT/${lang}-en/fbank2unit \
    --output-dir $CVSS_ROOT/${lang}-en/src_unigram200 \
    --lang $lang \
    --is-src-text \
    --vocab-type unigram --vocab-size 200