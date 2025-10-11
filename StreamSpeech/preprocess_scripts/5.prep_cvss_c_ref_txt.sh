
lang=$1
CVSS_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_ref_txt.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-txt $CVSS_ROOT/${lang}-en/fbank2unit/$split.txt
done

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_ref_unit.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-unit $CVSS_ROOT/${lang}-en/fbank2unit/$split.unit
done

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_src_txt.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-txt $CVSS_ROOT/${lang}-en/fbank2unit/$split.src
done