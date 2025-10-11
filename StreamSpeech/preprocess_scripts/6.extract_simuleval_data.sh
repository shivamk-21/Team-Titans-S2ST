lang=$1
CVSS_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
COVOST2_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/covost2
ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech


PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_simuleval_data.py \
    --cvss-dir $CVSS_ROOT/${lang}-en \
    --covost2-dir $COVOST2_ROOT/${lang} \
    --out-dir $CVSS_ROOT/${lang}-en/simuleval 