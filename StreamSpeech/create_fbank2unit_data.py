#!/usr/bin/env python3
"""
Create fbank2unit data structure from our existing preprocessed data
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import shutil
import torchaudio
import numpy as np
from tqdm import tqdm

def read_km_file(km_file):
    """Read k-means quantized units from file"""
    units_dict = {}
    with open(km_file, 'r') as f:
        for line in f:
            if '|' in line:
                parts = line.strip().split('|', 1)
                audio_id = parts[0].replace('.wav', '')  # Remove .wav extension
                unit_seq = parts[1]
                units_dict[audio_id] = unit_seq
    return units_dict

def create_fbank2unit_structure(lang):
    """Create fbank2unit structure for training"""
    
    base_path = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en")
    covost_path = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/covost2/{lang}")
    output_path = base_path / "fbank2unit"
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Copy config files
    
    config_src = Path(f"/run/media/shivamk21/data/ML-Project/StreamSpeech/configs/{lang}-en")
    shutil.copy(config_src / "config_gcmvn.yaml", output_path / "config_gcmvn.yaml")
    shutil.copy(config_src / "config_mtl_asr_st_ctcst.yaml", output_path / "config_mtl_asr_st_ctcst.yaml")
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        print(f"Processing {split} split for {lang}-en...")
        
        # Read the basic TSV file (contains audio IDs)
        tsv_file = base_path / f"{split}.tsv"
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found, skipping...")
            continue
            
        with open(tsv_file, 'r') as f:
            audio_ids = [line.strip() for line in f.readlines()]
        
        # Read corresponding text files
        txt_file = base_path / "fbank2unit" / f"{split}.txt"  # English target text
        texts = {}
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i < len(audio_ids):
                        texts[audio_ids[i]] = line.strip()
        
        # Read source text files
        src_file = base_path / "fbank2unit" / f"{split}.src"  # Source language text
        src_texts = {}
        if src_file.exists():
            with open(src_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i < len(audio_ids):
                        src_texts[audio_ids[i]] = line.strip()
            
        # Read k-means units
        km_file = base_path / f"{split}.km1000"
        units_dict = {}
        if km_file.exists():
            units_dict = read_km_file(km_file)
            
        # Create manifest
        manifest_data = []
        
        for i, audio_id in enumerate(audio_ids):
            # Get source text
            if audio_id in src_texts:
                src_text = src_texts[audio_id]
            elif audio_id in texts:
                src_text = texts[audio_id]
            else:
                src_text = ""
                
            # Get target text (English)
            if audio_id in texts:
                tgt_text = texts[audio_id]
            else:
                tgt_text = ""
                
            # Get units for this audio ID
            if audio_id in units_dict:
                tgt_units = units_dict[audio_id]
            else:
                tgt_units = "63"  # Default unit
            
            # Estimate frame counts (these are placeholders)
            src_n_frames = len(tgt_units.split()) * 25  # Rough estimate
            tgt_n_frames = len(tgt_units.split())
            
            # Create audio path reference - this should point to the actual feature file in the zip
            src_audio_path = f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en/src_fbank80.zip:{audio_id}.npy:0"
            
            manifest_data.append({
                'id': audio_id,
                'src_audio': src_audio_path,
                'src_n_frames': src_n_frames,
                'src_text': src_text,
                'tgt_text': tgt_text,
                'tgt_audio': tgt_units,
                'tgt_n_frames': tgt_n_frames
            })
        
        # Save manifest
        df = pd.DataFrame(manifest_data)
        output_file = output_path / f"{split}.tsv"
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Created {output_file} with {len(manifest_data)} samples")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code (hi, ma, mr)")
    args = parser.parse_args()
    
    create_fbank2unit_structure(args.lang)
    print(f"fbank2unit structure created for {args.lang}-en")

if __name__ == "__main__":
    main()