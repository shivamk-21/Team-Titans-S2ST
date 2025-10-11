#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from pathlib import Path
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from examples.speech_to_speech.preprocessing.data_utils import (
    gen_config_yaml,
    load_units,
    process_units,
)
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    get_zip_manifest,
    save_df_to_tsv,
)
from fairseq.data.audio.audio_utils import convert_waveform
import torchaudio
import soundfile as sf
from tqdm import tqdm
import shutil

MANIFEST_COLUMNS = [
    "id",
    "src_audio",
    "src_n_frames",
    "src_text",
    "tgt_text",
    "tgt_audio",
    "tgt_n_frames",
]

def process_simple(args):
    """Simplified processing for our custom data format"""
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)
    
    src_type = "audio" if args.use_audio_input else "fbank"
    tgt_type = "unit"
    output_tsv_dir = output_root / f"{src_type}2{tgt_type}"
    output_tsv_dir.mkdir(exist_ok=True)

    source_root = output_root / ("src_flac" if args.use_audio_input else "src_fbank80")
    source_zip_path = output_root / f"{source_root.name}.zip"

    # Data paths
    covost_root = Path(args.covost_data_root) / args.src_lang
    cvss_root = Path(args.cvss_data_root) / f"{args.src_lang}-en"
    
    if not covost_root.is_dir():
        raise NotADirectoryError(f"{covost_root} does not exist")
    if not cvss_root.is_dir():
        raise NotADirectoryError(f"{cvss_root} does not exist")

    # Process source audio/features
    if source_zip_path.exists():
        print(f"{source_zip_path} exists.")
    else:
        print("Extracting source audio/features...")
        source_root.mkdir(exist_ok=True)
        
        # Read our custom data format
        for split in ["train", "dev", "test"]:
            print(f"Processing {split} split...")
            
            # Read our CoVoST-style data
            covost_tsv_path = covost_root / f"covost_v2.{args.src_lang}_en.tsv"
            if not covost_tsv_path.exists():
                print(f"Warning: {covost_tsv_path} not found, skipping {split}")
                continue
                
            covost_df = pd.read_csv(covost_tsv_path, sep='\t')
            # Filter by split
            split_df = covost_df[covost_df['split'] == split]
            
            # Read CVSS-C mapping
            cvss_tsv_path = cvss_root / f"{split}.tsv"
            if cvss_tsv_path.exists():
                with open(cvss_tsv_path, 'r') as f:
                    cvss_ids = [line.strip().split('\t')[0] for line in f.readlines() if line.strip()]
                # Filter to only include items that exist in both datasets
                split_df = split_df[split_df['path'].str.replace('.wav', '').isin(cvss_ids)]
            
            for _, row in tqdm(split_df.iterrows(), desc=f"Processing {split}"):
                audio_path = covost_root / "clips" / row['path']
                utt_id = row['path'].replace('.wav', '').replace('.mp3', '')
                
                if not audio_path.exists():
                    continue
                    
                # Load and convert audio
                waveform, sample_rate = torchaudio.load(audio_path)
                src_sample_rate = 16_000
                waveform, sample_rate = convert_waveform(
                    waveform,
                    sample_rate,
                    to_mono=True,
                    to_sample_rate=src_sample_rate,
                )
                
                if args.use_audio_input:
                    sf.write(
                        (source_root / f"{utt_id}.flac").as_posix(),
                        waveform.T.numpy(),
                        sample_rate,
                    )
                else:
                    features = extract_fbank_features(
                        waveform, sample_rate, source_root / f"{utt_id}.npy"
                    )

        print("ZIPing source audios/features...")
        create_zip(source_root, source_zip_path)
        shutil.rmtree(source_root)

    print("Fetching ZIP manifest...")
    src_audio_paths, src_audio_lengths = get_zip_manifest(
        source_zip_path,
        is_audio=args.use_audio_input,
    )

    # Generate TSV manifest
    print("Generating manifest...")
    for split in [ "dev", "test","train"]:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        
        # Read our CoVoST-style data
        covost_tsv_path = covost_root / f"covost_v2.{args.src_lang}_en.tsv"
        if not covost_tsv_path.exists():
            continue
            
        covost_df = pd.read_csv(covost_tsv_path, sep='\t')
        split_df = covost_df[covost_df['split'] == split]
        
        # Read CVSS-C mapping
        cvss_tsv_path = cvss_root / f"{split}.tsv"
        if cvss_tsv_path.exists():
            with open(cvss_tsv_path, 'r') as f:
                cvss_ids = [line.strip().split('\t')[0] for line in f.readlines() if line.strip()]
            split_df = split_df[split_df['path'].str.replace('.wav', '').isin(cvss_ids)]
        
        # Load target units
        target_unit_data = load_units(cvss_root / f"{split}.{args.unit_type}")
        
        # Read source texts from our short.tsv
        short_tsv_path = cvss_root / f"{split}.short.tsv"
        src_texts = {}
        if short_tsv_path.exists():
            short_df = pd.read_csv(short_tsv_path, sep='\t')
            for _, row in short_df.iterrows():
                src_texts[row['audio_id']] = row['src_text']
        
        for _, row in tqdm(split_df.iterrows(), desc=f"Creating manifest for {split}"):
            utt_id = row['path'].replace('.wav', '').replace('.mp3', '')
            
            if utt_id not in src_audio_paths:
                continue
                
            manifest["id"].append(utt_id)
            manifest["src_audio"].append(src_audio_paths[utt_id])
            manifest["src_n_frames"].append(src_audio_lengths[utt_id])
            manifest["src_text"].append(src_texts.get(utt_id, ""))
            manifest["tgt_text"].append(row['translation'])
            
            # Process target units
            target_key = f"{utt_id}.wav"
            if target_key in target_unit_data:
                target_units = process_units(target_unit_data[target_key], args.reduce_unit)
                manifest["tgt_audio"].append(" ".join(target_units))
                manifest["tgt_n_frames"].append(len(target_units))
            else:
                manifest["tgt_audio"].append("")
                manifest["tgt_n_frames"].append(0)

        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, output_tsv_dir / f"{split}.tsv")

    # Generate config YAML
    if args.use_audio_input:
        gen_config_yaml(
            output_tsv_dir,
            specaugment_policy=None,
            feature_transform=["utterance_cmvn"],
            vocoder_type="code_hifigan",
            vocoder_checkpoint=args.vocoder_checkpoint,
            vocoder_cfg=args.vocoder_cfg,
            extra={"use_audio_input": True},
        )
    else:
        gen_config_yaml(
            output_tsv_dir,
            specaugment_policy="lb",
            feature_transform=["utterance_cmvn"],
            vocoder_type="code_hifigan",
            vocoder_checkpoint=args.vocoder_checkpoint,
            vocoder_cfg=args.vocoder_cfg,
        )
    print(output_tsv_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvss-data-root", required=True, type=str)
    parser.add_argument("--covost-data-root", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    parser.add_argument("--use-audio-input", action="store_true")
    parser.add_argument("--src-lang", required=True, type=str)
    parser.add_argument("--unit-type", default="km1000", type=str)
    parser.add_argument("--reduce-unit", action="store_true")
    parser.add_argument("--vocoder-checkpoint", default=None, type=str)
    parser.add_argument("--vocoder-cfg", default=None, type=str)

    args = parser.parse_args()
    process_simple(args)

if __name__ == "__main__":
    main()