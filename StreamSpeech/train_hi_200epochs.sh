#!/bin/bash

# Individual Hindi training script for 200 epochs
# Enhanced with comprehensive logging and checkpoint retention

set -e

export CUDA_VISIBLE_DEVICES=0

LANG=$1
EPOCHS=$2
DATA_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
DATA=$DATA_ROOT/${LANG}-en/fbank2unit
model=streamspeech.offline-s2st.${LANG}-en

# Create log directories
LOG_DIR="/run/media/shivamk21/data/ML-Project/StreamSpeech/training/training_logs"
TENSORBOARD_DIR="/run/media/shivamk21/data/ML-Project/StreamSpeech/training/tensorboard_logs"
mkdir -p $LOG_DIR
mkdir -p $TENSORBOARD_DIR

# Log files 
log_file="$LOG_DIR/train_${LANG}_${EPOCHS}epochs_$(date +%Y%m%d_%H%M%S).log"
tensorboard_log="$TENSORBOARD_DIR/${LANG}_${EPOCHS}epochs"

echo "🇮🇳 Starting $LANG training for $EPOCHS epochs"
echo "📁 Data: $DATA"
echo "📝 Log: $log_file"
echo "📊 Tensorboard: $tensorboard_log"
echo "💾 Checkpoints: checkpoints/$model"

# Activate conda environment
echo "🐍 Activating conda environment..."
source ~/.bashrc
conda activate speech
export PYTHONPATH=/run/media/shivamk21/data/ML-Project/StreamSpeech/fairseq:$PYTHONPATH

fairseq-train $DATA \
  --user-dir /run/media/shivamk21/data/ML-Project/StreamSpeech/researches/ctc_unity \
  --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
  --task speech_to_speech_ctc --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --criterion speech_to_unit_2pass_ctc_asr_st --label-smoothing 0.1 --rdrop-alpha 0.0 \
  --arch streamspeech --share-decoder-input-output-embed \
  --encoder-layers 12 --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4 \
  --translation-decoder-layers 4 --synthesizer-encoder-layers 2 \
  --decoder-layers 2 --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 8 \
  --k1 0 --k2 0 --n1 1 --n2 -1 \
  --chunk-size 8000 \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --ctc-upsample-rate 25 \
  --save-dir /run/media/shivamk21/data/ML-Project/StreamSpeech/training/checkpoints/$model \
  --validate-interval 1000 --validate-interval-updates 1000 \
  --save-interval 50 --save-interval-updates 1000 \
  --keep-last-epochs -1 \
  --max-epoch $EPOCHS \
  --no-progress-bar --log-format json --log-interval 50 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 15000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 1.0 \
  --max-tokens 5000 --max-source-positions 50000 --max-target-positions 512 --update-freq 2 \
  --skip-invalid-size-inputs-valid-test \
  --attn-type espnet --pos-enc-type rel_pos \
  --keep-interval-updates -1 \
  --keep-best-checkpoints -1 \
  --tensorboard-logdir $tensorboard_log \
  --seed 1 --fp16 --num-workers 1 \
  --empty-cache-freq 100 \
  2>&1 | tee $log_file

echo "✅ $LANG training completed!"
echo "📝 Log saved to: $log_file"
echo "📊 Tensorboard logs: $tensorboard_log"