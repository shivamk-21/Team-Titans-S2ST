#!/usr/bin/env python3
"""
Fix fbank2unit data by creating proper ZIP manifest format
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import zipfile
import numpy as np

def get_zip_manifest(zip_path):
    """Get file byte offsets from ZIP file"""
    manifest = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        offset = 0
        for info in zip_file.infolist():
            if info.filename.endswith('.npy'):
                # Get the compressed size and offset
                file_id = info.filename.replace('.npy', '')
                # For stored files (no compression), compressed_size = file_size
                manifest[file_id] = {
                    'offset': info.header_offset + 30 + len(info.filename) + len(info.extra),
                    'length': info.file_size
                }
    
    return manifest

def fix_fbank2unit_structure(lang):
    """Fix fbank2unit structure with correct ZIP paths"""
    
    base_path = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en")
    zip_path = base_path / "src_fbank80.zip"
    fbank2unit_path = base_path / "fbank2unit"
    
    # Get ZIP manifest
    manifest = get_zip_manifest(zip_path)
    print(f"Got manifest for {len(manifest)} files")
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        tsv_file = fbank2unit_path / f"{split}.tsv"
        if not tsv_file.exists():
            continue
            
        print(f"Processing {split} split...")
        
        # Read existing TSV
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Fix src_audio paths
        fixed_paths = []
        for idx, row in df.iterrows():
            audio_id = row['id']
            if audio_id in manifest:
                offset = manifest[audio_id]['offset']
                length = manifest[audio_id]['length']
                fixed_path = f"{zip_path}:{offset}:{length}"
            else:
                # Fallback: use simple path
                fixed_path = f"{zip_path}:{audio_id}.npy"
                print(f"Warning: {audio_id} not found in manifest, using fallback")
            
            fixed_paths.append(fixed_path)
        
        df['src_audio'] = fixed_paths
        
        # Save fixed TSV
        df.to_csv(tsv_file, sep='\t', index=False)
        print(f"Fixed {len(df)} samples in {tsv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code (hi, ma, mr)")
    args = parser.parse_args()
    
    fix_fbank2unit_structure(args.lang)
    print(f"Fixed fbank2unit structure for {args.lang}-en")

if __name__ == "__main__":
    main()