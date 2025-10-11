#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Check if language parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <language>"
    echo "Available languages: hi, ma, mr"
    echo "Example: $0 hi"
    exit 1
fi

LANG=$1

# Validate language parameter
if [[ "$LANG" != "hi" && "$LANG" != "ma" && "$LANG" != "mr" ]]; then
    echo "Error: Unsupported language '$LANG'"
    echo "Supported languages: hi (Hindi), ma (Malayalam), mr (Marathi)"
    exit 1
fi

# Ensure we're using the correct conda environment
eval "$(conda shell.bash hook)"
conda activate speech

ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech
DATA_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c
PRETRAIN_ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech/pretrained_models
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json
DATA=${DATA_ROOT}/${LANG}-en/fbank2unit
SPLIT=test
BEAM=1
export PYTHONPATH=/run/media/shivamk21/data/ML-Project/StreamSpeech/fairseq:$PYTHONPATH
file=/run/media/shivamk21/data/ML-Project/StreamSpeech/training/checkpoints/streamspeech.offline-s2st.${LANG}-en/checkpoint_best.pt

# Validate that the checkpoint file exists
if [ ! -f "$file" ]; then
    echo "Error: Checkpoint file not found: $file"
    echo "Available checkpoints:"
    ls -la /run/media/shivamk21/data/ML-Project/StreamSpeech/training/checkpoints/ 2>/dev/null || echo "No checkpoints directory found"
    exit 1
fi

# Validate that the data directory exists
if [ ! -d "$DATA" ]; then
    echo "Error: Data directory not found: $DATA"
    echo "Available data directories:"
    ls -la ${DATA_ROOT}/ 2>/dev/null || echo "No data root directory found"
    exit 1
fi

cd /run/media/shivamk21/data/ML-Project
mkdir -p res
output_dir=/run/media/shivamk21/data/ML-Project/res/streamspeech.offline.${LANG}-en
mkdir -p $output_dir

echo "========================================="
echo "StreamSpeech Offline S2ST Evaluation"
echo "========================================="
echo "Language: $LANG -> en"
echo "Checkpoint: $file"
echo "Data directory: $DATA"
echo "Output directory: $output_dir"
echo "Timestamp: $(date)"
echo "========================================="

echo "Step 1/6: Generating model outputs..."
cd $ROOT
PYTHONPATH=$ROOT/fairseq:$PYTHONPATH python fairseq/fairseq_cli/generate.py ${DATA} \
    --user-dir researches/ctc_unity \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --task speech_to_speech_ctc --target-is-code --target-code-size 1000 --vocoder code_hifigan \
    --path $file --gen-subset $SPLIT \
    --beam-mt $BEAM --beam 1 --max-len-a 1 \
    --max-tokens 10000 \
    --required-batch-size-multiple 1 \
    --skip-invalid-size-inputs-valid-test \
    --results-path $output_dir > $output_dir/generate-$SPLIT.log 2>&1

# Check if generation was successful
if [ ! -f "$output_dir/generate-$SPLIT.log" ] || [ ! -s "$output_dir/generate-$SPLIT.log" ]; then
    echo "Error: Generation failed or produced no output"
    echo "Check the log file: $output_dir/generate-$SPLIT.log"
    if [ -f "$output_dir/generate-$SPLIT.log" ]; then
        echo "Log file contents:"
        cat $output_dir/generate-$SPLIT.log
    fi
    exit 1
fi

echo "Step 2/6: Extracting and aligning ASR hypotheses..."
# Extract ASR hypotheses
grep '^A-' $output_dir/generate-$SPLIT.log | sort -t'-' -k2,2n | cut -f2 > $output_dir/generate-$SPLIT.asr
if [ ! -s "$output_dir/generate-$SPLIT.asr" ]; then
    echo "Warning: No ASR hypotheses found in output"
fi

# Extract sample IDs and align reference with hypotheses properly
python - <<END
import os

ref_file = "$DATA/$SPLIT.src"
hyp_file = "$output_dir/generate-$SPLIT.asr"
log_file = "$output_dir/generate-$SPLIT.log"
out_hyp_file = "$output_dir/generate-$SPLIT.asr.aligned"
out_ref_file = "$output_dir/generate-$SPLIT.src.aligned"

# Read references
with open(ref_file) as f:
    references = f.read().splitlines()

# Read hypotheses with sample IDs from log
hyp_with_ids = []
with open(log_file) as f:
    for line in f:
        if line.startswith("A-"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sample_id = int(parts[0].split("-")[1])
                hypothesis = parts[1] if len(parts) > 1 else ""
                hyp_with_ids.append((sample_id, hypothesis))

# Sort by sample ID
hyp_with_ids.sort()

# Align references and hypotheses based on available sample IDs
aligned_refs = []
aligned_hyps = []

for sample_id, hypothesis in hyp_with_ids:
    if sample_id < len(references):
        aligned_refs.append(references[sample_id])
        aligned_hyps.append(hypothesis)
    else:
        # If sample_id exceeds reference length, skip this sample
        print(f"Warning: Sample ID {sample_id} exceeds reference length ({len(references)}), skipping")

# Write aligned files
with open(out_ref_file, "w") as f:
    for line in aligned_refs:
        f.write(line + "\n")

with open(out_hyp_file, "w") as f:
    for line in aligned_hyps:
        f.write(line + "\n")

print(f"Aligned {len(aligned_refs)} samples")
END

echo "Step 3/6: Computing ASR BLEU and WER scores..."
# Compute BLEU for ASR source text
echo '################### ASR source text BLEU ###################' >> $output_dir/res.txt
sacrebleu $output_dir/generate-$SPLIT.src.aligned -i $output_dir/generate-$SPLIT.asr.aligned -w 3 >> $output_dir/res.txt

# Compute WER safely
python - <<END >> $output_dir/res.txt
from jiwer import wer

with open("$output_dir/generate-$SPLIT.src.aligned") as f:
    references = f.read().splitlines()
with open("$output_dir/generate-$SPLIT.asr.aligned") as f:
    hypotheses = f.read().splitlines()

if len(references) == len(hypotheses) and len(references) > 0:
    error = wer(references, hypotheses)
    print("WER:", error)
else:
    print(f"WER: Cannot compute - mismatched lengths (ref: {len(references)}, hyp: {len(hypotheses)})")
END


echo "Step 4/6: Processing target text and unit outputs..."
# Extract and align target text
python - <<END
import os

tgt_ref_file = "$DATA/$SPLIT.txt"
log_file = "$output_dir/generate-$SPLIT.log"
out_tgt_file = "$output_dir/generate-$SPLIT.tgt"
out_tgt_ref_file = "$output_dir/generate-$SPLIT.tgt.ref.aligned"

# Read target references
with open(tgt_ref_file) as f:
    tgt_references = f.read().splitlines()

# Read target hypotheses with sample IDs from log
tgt_with_ids = []
with open(log_file) as f:
    for line in f:
        if line.startswith("D-"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sample_id = int(parts[0].split("-")[1])
                target = parts[1] if len(parts) > 1 else ""
                tgt_with_ids.append((sample_id, target))

# Sort by sample ID
tgt_with_ids.sort()

# Align target references and hypotheses
aligned_tgt_refs = []
aligned_tgt_hyps = []

for sample_id, target in tgt_with_ids:
    if sample_id < len(tgt_references):
        aligned_tgt_refs.append(tgt_references[sample_id])
        aligned_tgt_hyps.append(target)

# Write aligned files
with open(out_tgt_ref_file, "w") as f:
    for line in aligned_tgt_refs:
        f.write(line + "\n")

with open(out_tgt_file, "w") as f:
    for line in aligned_tgt_hyps:
        f.write(line + "\n")

print(f"Aligned {len(aligned_tgt_refs)} target samples")
END

if [ ! -s "$output_dir/generate-$SPLIT.tgt" ]; then
    echo "Warning: No target text found in output"
fi

echo '################### Speech-to-text target text BLEU ###################' >> $output_dir/res.txt
sacrebleu $output_dir/generate-$SPLIT.tgt.ref.aligned -i $output_dir/generate-$SPLIT.tgt -w 3 >> $output_dir/res.txt

# Extract and align unit outputs
python - <<END
import os

unit_ref_file = "$DATA/$SPLIT.unit"
log_file = "$output_dir/generate-$SPLIT.log"
out_unit_file = "$output_dir/generate-$SPLIT.unit"
out_unit_ref_file = "$output_dir/generate-$SPLIT.unit.ref.aligned"

# Read unit references
with open(unit_ref_file) as f:
    unit_references = f.read().splitlines()

# Read unit hypotheses with sample IDs from log
# Try multiple possible formats: H-, U-, T- for units
unit_with_ids = []
unit_prefixes = ["H-", "U-", "T-"]

with open(log_file) as f:
    for line in f:
        for prefix in unit_prefixes:
            if line.startswith(prefix):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    sample_id = int(parts[0].split("-")[1])
                    unit = parts[2] if len(parts) > 2 else ""
                    unit_with_ids.append((sample_id, unit))
                break

# If no units found, create empty files
if not unit_with_ids:
    print("Warning: No unit outputs found in log. Creating empty unit files.")
    with open(out_unit_file, "w") as f:
        pass  # Create empty file
    with open(out_unit_ref_file, "w") as f:
        pass  # Create empty file
    print(f"Aligned 0 unit samples")
else:
    # Sort by sample ID
    unit_with_ids.sort()

    # Align unit references and hypotheses
    aligned_unit_refs = []
    aligned_unit_hyps = []

    for sample_id, unit in unit_with_ids:
        if sample_id < len(unit_references):
            aligned_unit_refs.append(unit_references[sample_id])
            aligned_unit_hyps.append(unit)

    # Write aligned files
    with open(out_unit_ref_file, "w") as f:
        for line in aligned_unit_refs:
            f.write(line + "\n")

    with open(out_unit_file, "w") as f:
        for line in aligned_unit_hyps:
            f.write(line + "\n")

    print(f"Aligned {len(aligned_unit_refs)} unit samples")
END

echo '################### Speech-to-unit target unit BLEU ###################' >> $output_dir/res.txt
sacrebleu $output_dir/generate-$SPLIT.unit.ref.aligned -i $output_dir/generate-$SPLIT.unit -w 3 >> $output_dir/res.txt

echo "Step 5/6: Generating waveforms from units..."
if [ -s "$output_dir/generate-$SPLIT.unit" ]; then
    python $ROOT/fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
        --in-code-file $output_dir/generate-$SPLIT.unit \
        --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
        --results-path $output_dir/pred_wav --dur-prediction
else
    echo "Warning: No units available for waveform generation. Skipping audio synthesis."
    mkdir -p $output_dir/pred_wav
fi

echo "Step 6/6: Computing ASR-BLEU on generated speech..."
if [ -d "$output_dir/pred_wav" ] && [ "$(ls -A $output_dir/pred_wav 2>/dev/null | head -1)" ]; then
    cd $ROOT/asr_bleu
    python compute_asr_bleu.py \
        --lang en \
        --audio_dirpath $output_dir/pred_wav \
        --reference_path $output_dir/generate-$SPLIT.tgt.ref.aligned \
        --reference_format txt > $output_dir/asr_bleu.log 2>&1

    cd $ROOT

    echo '################### Speech-to-speech target speech ASR-BLEU ###################' >> $output_dir/res.txt
    if [ -s "$output_dir/asr_bleu.log" ]; then
        tail -n 1 $output_dir/asr_bleu.log >> $output_dir/res.txt
    else
        echo "ASR-BLEU: Error in computation - see asr_bleu.log" >> $output_dir/res.txt
    fi
else
    echo '################### Speech-to-speech target speech ASR-BLEU ###################' >> $output_dir/res.txt
    echo "ASR-BLEU: Skipped - no synthesized audio files available" >> $output_dir/res.txt
fi

echo "========================================="
echo "✓ Evaluation completed successfully!"
echo "Language: $LANG -> en"
echo "Results saved in: $output_dir/res.txt"
echo "========================================="
echo ""
echo "RESULTS SUMMARY:"
cat $output_dir/res.txt