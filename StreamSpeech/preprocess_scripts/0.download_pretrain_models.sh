ROOT=/run/media/shivamk21/data/ML-Project/StreamSpeech
PRETRAIN_ROOT=$ROOT/pretrained_models

# Create directories
mkdir -p $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder
mkdir -p $PRETRAIN_ROOT/mHuBERT

# ---------------------------
# HuBERT model
# ---------------------------
# Using mHuBERT-147 (supports Hindi, Marathi, Malayalam)
if [ ! -f "$PRETRAIN_ROOT/mHuBERT/pytorch_model.bin" ]; then
    wget -P $PRETRAIN_ROOT/mHuBERT https://huggingface.co/utter-project/mHuBERT-147/resolve/main/pytorch_model.bin
fi

if [ ! -f "$PRETRAIN_ROOT/mHuBERT/config.json" ]; then
    wget -P $PRETRAIN_ROOT/mHuBERT https://huggingface.co/utter-project/mHuBERT-147/resolve/main/config.json
fi

if [ ! -f "$PRETRAIN_ROOT/mHuBERT/mhubert_base_vp_147lang.pt" ]; then
    wget -O $PRETRAIN_ROOT/mHuBERT/mhubert_base_vp_147lang.pt https://huggingface.co/utter-project/mHuBERT-147/resolve/main/checkpoint_best.pt
fi

# ---------------------------
# Vocoders
# ---------------------------
mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder 
mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en 
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json
