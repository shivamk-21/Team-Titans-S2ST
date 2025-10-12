#!/usr/bin/env python3
"""
Create complete config directories for each language
This creates all missing files in configs/<lang>-en/ directories
"""

import os
import numpy as np
from pathlib import Path
import argparse
import shutil

def create_gcmvn_files(lang, config_dir):
    """Create gcmvn.npz and config_gcmvn.yaml"""
    
    # Create dummy GCMVN stats (you can run create_gcmvn.py later for real stats)
    gcmvn_path = config_dir / "gcmvn.npz"
    
    # Create dummy statistics
    mean = np.zeros(80)  # 80-dim fbank features
    std = np.ones(80)
    
    np.savez(gcmvn_path, mean=mean, std=std)
    print(f"Created GCMVN file: {gcmvn_path}")
    
    # Create config_gcmvn.yaml
    cfg_path = config_dir / "config_gcmvn.yaml"
    cfg_text = f"""global_cmvn:
  stats_npz_path: {gcmvn_path.as_posix()}
input_channels: 1
input_feat_per_channel: 80
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment
vocoder:
  checkpoint: /run/media/shivamk21/data/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
  config: /run/media/shivamk21/data/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json
  type: code_hifigan
"""
    
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(cfg_text)
    print(f"Created config_gcmvn.yaml: {cfg_path}")

def create_vocab_dirs(lang, config_dir):
    """Create src_unigram200 and tgt_unigram200 directories with dummy vocab files"""
    # This function will only create placeholder unigram dirs if they don't
    # already exist. If processed_datasets contain unigram files we prefer to
    # copy them (handled in create_all_configs_for_language).
    for vocab_type in ['src_unigram200', 'tgt_unigram200']:
        vocab_dir = config_dir / vocab_type
        if not vocab_dir.exists():
            vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = vocab_dir / f"spm_unigram_{lang}.txt"
        model_file = vocab_dir / f"spm_unigram_{lang}.model"

        if vocab_file.exists():
            print(f"Skipping existing vocab file: {vocab_file}")
            continue

        # Create a minimal placeholder vocabulary (space-separated, no special tokens)
        vocab_lines = [
            "▁ 4",
            "a 5",
            "e 6",
            "i 7",
            "o 8",
            "u 9",
        ]
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(vocab_lines) + "\n")

        # Create empty model file placeholder if not present
        if not model_file.exists():
            model_file.touch()

        print(f"Created placeholder vocab directory: {vocab_dir}")

def create_multitask_config(lang, config_dir):
    """Create config_mtl_asr_st_ctcst.yaml"""
    
    cfg_path = config_dir / "config_mtl_asr_st_ctcst.yaml"
    # Prefer unigram files from the processed_datasets location if available
    processed_unigram_base = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en")
    tgt_unigram_dir = processed_unigram_base / "tgt_unigram200"
    src_unigram_dir = processed_unigram_base / "src_unigram200"
    # fall back to local config_dir if processed datasets are not present
    tgt_dict_path = (tgt_unigram_dir / f"spm_unigram_{lang}.txt").as_posix() if tgt_unigram_dir.exists() else (config_dir / "tgt_unigram200" / f"spm_unigram_{lang}.txt").as_posix()
    tgt_data_path = tgt_unigram_dir.as_posix() if tgt_unigram_dir.exists() else (config_dir / "tgt_unigram200").as_posix()
    src_dict_path = (src_unigram_dir / f"spm_unigram_{lang}.txt").as_posix() if src_unigram_dir.exists() else (config_dir / "src_unigram200" / f"spm_unigram_{lang}.txt").as_posix()
    src_data_path = src_unigram_dir.as_posix() if src_unigram_dir.exists() else (config_dir / "src_unigram200").as_posix()

        # Properly indented YAML for multitask config
    cfg_text = f"""target_unigram:
    decoder_type: transformer
    dict: {tgt_dict_path}
    data: {tgt_data_path}
    loss_weight: 8.0
    rdrop_alpha: 0.0
    decoder_args:
        decoder_layers: 4
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
    label_smoothing: 0.1
source_unigram:
    decoder_type: ctc
    dict: {src_dict_path}
    data: {src_data_path}
    loss_weight: 4.0
    rdrop_alpha: 0.0
    decoder_args:
        decoder_layers: 0
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
    label_smoothing: 0.1
ctc_target_unigram:
    decoder_type: ctc
    dict: {tgt_dict_path}
    data: {tgt_data_path}
    loss_weight: 4.0
    rdrop_alpha: 0.0
    decoder_args:
        decoder_layers: 0
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
    label_smoothing: 0.1
"""
    
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(cfg_text)
    print(f"Created config_mtl_asr_st_ctcst.yaml: {cfg_path}")

def create_unity_config(lang, config_dir):
    """Create config_unity.yaml"""
    
    cfg_path = config_dir / "config_unity.yaml"
    processed_unigram_base = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en")
    tgt_unigram_dir = processed_unigram_base / "tgt_unigram200"
    tgt_dict_path = (tgt_unigram_dir / f"spm_unigram_{lang}.txt").as_posix() if tgt_unigram_dir.exists() else (config_dir / "tgt_unigram200" / f"spm_unigram_{lang}.txt").as_posix()
    tgt_data_path = tgt_unigram_dir.as_posix() if tgt_unigram_dir.exists() else (config_dir / "tgt_unigram200").as_posix()

    cfg_text = f"""target_unigram:
    decoder_type: transformer
    dict: {tgt_dict_path}
    data: {tgt_data_path}
    loss_weight: 8.0
    rdrop_alpha: 0.0
    decoder_args:
        decoder_layers: 4
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
    label_smoothing: 0.1
"""
    
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(cfg_text)
    print(f"Created config_unity.yaml: {cfg_path}")

def create_all_configs_for_language(lang, overwrite=False):
    """Create complete config directory for a language"""
    
    config_dir = Path(f"/run/media/shivamk21/data/ML-Project/StreamSpeech/configs/{lang}-en")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔧 Creating configs for {lang}-en in {config_dir}")

    # Prefer and copy unigram files from processed_datasets if they exist
    processed_unigram_base = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en")
    for side in ["tgt_unigram200", "src_unigram200"]:
        src_dir = processed_unigram_base / side
        dest_dir = config_dir / side
        if src_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            # copy ALL files from the processed unigram directory into config dir
            for s in src_dir.iterdir():
                if not s.is_file():
                    continue
                d = dest_dir / s.name
                if d.exists():
                    if overwrite:
                        shutil.copy2(s, d)
                        print(f"OVERWRITTEN {s} -> {d}")
                    else:
                        print(f"SKIP {s} -> {d} (exists)")
                    continue
                shutil.copy2(s, d)
                print(f"COPIED {s} -> {d}")

    # Create components (create_vocab_dirs will skip existing files)
    create_gcmvn_files(lang, config_dir)
    create_vocab_dirs(lang, config_dir)
    create_multitask_config(lang, config_dir)
    create_unity_config(lang, config_dir)
    
    print(f"✅ Completed config creation for {lang}-en")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="Language code (hi, ma, mr) or 'all' for all languages")
    parser.add_argument("--languages", nargs='+', default=['hi', 'ma', 'mr'], 
                       help="List of language codes to process")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files when copying from processed_datasets")
    args = parser.parse_args()
    
    if args.lang == 'all' or args.lang is None:
        languages = args.languages
    else:
        languages = [args.lang]
    
    print("🎯 Creating complete config directories for all languages")
    print("=" * 60)
    
    for lang in languages:
        create_all_configs_for_language(lang, overwrite=args.overwrite)
    
    print(f"\n🎉 All config directories created!")
    print("\nNext steps:")
    print("1. Run create_gcmvn.py --lang <lang> to replace dummy GCMVN with real stats")
    print("2. Run create_fbank2unit_data.py --lang <lang> to copy configs to dataset")
    print("3. Generate proper vocab files using preprocessing scripts if needed")

if __name__ == "__main__":
    main()