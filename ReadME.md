## TeamTitans : Speech-to-Speech (S2S) workflow for Indic Languages

This repository contains code and data preparation scripts for speech-to-speech experiments built on top of StreamSpeech and a modified Fairseq. The README below focuses on the speech-to-speech (S2S) pipeline and gives an "Overall running code" checklist you can follow to reproduce training and preprocessing steps.

## Assumptions and quick notes
- Python 3.8+ and CUDA (optional, recommended for training) are available.
- PyTorch and other Python dependencies should be installed in a virtualenv/conda env.
- The repository root contains these important folders: `StreamSpeech/`, `fairseq/`, `preprocess_scripts/`, `Datasets/`, `configs/`.
- The instructions below assume language IDs like `hi-en`, `ma-en`, `mr-en` (replace with the language pair you want to run).

## Overall running code (end-to-end steps)
Follow these steps in order. Replace `<LANG>` with your language pair, for example `hi-en`, `ma-en`, or `mr-en`.

1) Clone StreamSpeech

If you don't already have the StreamSpeech project, clone it. If you already have the `StreamSpeech/` folder in this repo, you can skip cloning and use the local copy.

```bash
git clone https://github.com/ictnlp/StreamSpeech/tree/main
```

2) Replace / modify fairseq with our modified version

Follow StreamSpeech Library Installlation Guidelines

3) Run preprocessing (preprocess.sh) with the language

We use the provided preprocessing scripts to prepare features and text for training. There are multiple helper scripts in `preprocess_scripts/` and/or in `StreamSpeech/`  use the one that matches your pipeline. Example (replace `<LANG>`):

```bash
# Example script — change if you have a specific preprocess.sh in StreamSpeech
bash preprocess_scripts/2.prep_cvss_c_multilingual_data.sh "$LANG"  # e.g. hi
```

Notes:
- If the StreamSpeech distribution contains its own `preprocess.sh` or data prep helper, prefer that script and pass the same `<LANG>` variable.
- Preprocessing will populate `Datasets/processed_datasets/` or the `StreamSpeech/data/` folder depending on the script.

4) Run `create_configs.py` to generate model/config files

This repository includes `StreamSpeech/create_configs.py` which prepares training configs. Run it with the language pair to create per-language config files.

```bash
python3 ./StreamSpeech/create_configs.py --lang "$LANG_PAIR"
# This writes config files to `configs/<LANG_PAIR>/` (or prints where it writes them). Inspect and edit params like batch size, learning rate, and checkpoints.
```

5) Generate global CMVN stats

Run the global cepstral mean-variance normalization (gcmvn) generator to compute normalization stats for the features.

```bash
python3 ./StreamSpeech/create_gcmvn.py --lang "$LANG_PAIR"
```

This creates mean/variance files used by the feature extraction pipeline.

6) Run fbank2unit data prep

Use the fbank-to-unit data script to create final training data in the format expected by StreamSpeech/fairseq.

```bash
python3 ./StreamSpeech/create_fbank2unit_data.py --lang "$LANG_PAIR"
```

After this step you should have feature directories / manifests that the trainer will consume (check `data/` or `Datasets/processed_datasets/` paths).

7) Run training

Start training with the generated config.
```bash
bash StreamSpeech/train_hi_200epochs.sh <lang> <epochs> # Like hi 30
```
or
```bash
bash StreamSpeech/train_hi_memory_optimized.sh <epochs> # Like hi 30
```

Check `Results/` for outputs (for example there are example offline result directories at `Results/res/streamspeech.offline.<LANG_PAIR>/`).


## Files/locations of interest
- `StreamSpeech/` — main S2S code, helper scripts, and training wrappers.
- `preprocess_scripts/` — dataset-specific preprocessing helpers used to prepare features.
- `configs/` — generated configs per language pair.
- `Dataset` - Contain all type of datasets (original, preocessed)
- `Datasets/processed_datasets/` — prepped data folds.
- `Results/` — example outputs and trained model directories.


## Quality gates & verification
- Can run `/run/media/shivamk21/E9DF-F069/TeamTitans/StreamSpeech/researches/ctc_unity/test_scripts/pred.offline-s2st.sh`

## Quick Links
- Original Paper : https://arxiv.org/pdf/2406.03049
- Original Repo : https://github.com/ictnlp/StreamSpeech/tree/main
- Original Datset : https://huggingface.co/collections/ai4bharat/bhasaanuvaad-672b3790b6470eab68b1cb87 
- Speech to Text Dataset : https://drive.google.com/file/d/1uF2ZSM7CdXaRG6c8fmAlh7qMWcNTFktx
