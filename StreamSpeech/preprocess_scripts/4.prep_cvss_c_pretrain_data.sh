lang=$1
CVSS_ROOT=/run/media/shivamk21/data/ML-Project/processed_datasets/cvss/cvss-c
ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/convert_s2st_tsv_to_s2tt_tsv.py \
    --s2st-tsv-dir $CVSS_ROOT/${lang}-en/fbank2unit \
    --s2tt-tsv-dir $CVSS_ROOT/${lang}-en/fbank2text  \

cp $CVSS_ROOT/${lang}-en/tgt_unigram200/spm* $CVSS_ROOT/${lang}-en/fbank2text 