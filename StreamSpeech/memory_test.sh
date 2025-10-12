#!/bin/bash

# Quick memory test - just 1 epoch to check if optimizations work
# This will help us validate the memory optimization strategy

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

LANG=hi
EPOCHS=1
DATA_ROOT=/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/
DATA=$DATA_ROOT/${LANG}-en/fbank2unit
model=streamspeech.offline-s2st.${LANG}-en.test

LOG_DIR="/run/media/shivamk21/data/ML-Project/StreamSpeech/training/training_logs"
mkdir -p $LOG_DIR
log_file="$LOG_DIR/memory_test_$(date +%Y%m%d_%H%M%S).log"

echo "🧪 MEMORY TEST - Starting minimal training to test memory optimization"
echo "📝 Log: $log_file"

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cleared')" 2>/dev/null || echo "PyTorch not available in base env"

source ~/.bashrc
conda activate speech
export PYTHONPATH=/run/media/shivamk21/data/ML-Project/StreamSpeech/fairseq:$PYTHONPATH

# Test GPU memory
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Total memory: {props.total_memory / 1e9:.2f} GB')
    print(f'Memory available: {(props.total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB')
    torch.cuda.empty_cache()
else:
    print('CUDA not available')
"

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
  --chunk-size 4000 \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --ctc-upsample-rate 25 \
  --save-dir /tmp/test_checkpoints \
  --validate-interval 5000 --validate-interval-updates 5000 \
  --save-interval 5000 --save-interval-updates 5000 \
  --keep-last-epochs 1 \
  --max-epoch 1 \
  --max-update 10 \
  --no-progress-bar --log-format json --log-interval 5 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 5 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.5 \
  --max-tokens 2500 --max-source-positions 30000 --max-target-positions 300 --update-freq 1 \
  --skip-invalid-size-inputs-valid-test \
  --attn-type espnet --pos-enc-type rel_pos \
  --tensorboard-logdir /tmp/test_tb \
  --seed 1 --num-workers 0 \
  --empty-cache-freq 5 \
  --distributed-no-spawn \
  2>&1 | tee $log_file

echo "🧪 Memory test completed!"
echo "📝 Log: $log_file"