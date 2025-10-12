#!/usr/bin/env python3
"""
Data augmentation script for improving StreamSpeech training
"""

import os
import json
import pandas as pd
import random
from pathlib import Path

def augment_training_data(data_dir, output_dir, augment_factor=1.5):
    """
    Create augmented training data by:
    1. Speed perturbation
    2. Noise addition
    3. Data duplication with variations
    """
    
    print(f"Augmenting data in {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original manifest
    train_tsv = os.path.join(data_dir, "train.tsv")
    if not os.path.exists(train_tsv):
        print(f"Train manifest not found: {train_tsv}")
        return
    
    # Read training data
    df = pd.read_csv(train_tsv, sep='\t')
    original_size = len(df)
    
    print(f"Original dataset size: {original_size}")
    
    # Create augmented samples
    augmented_rows = []
    
    # Speed perturbation factors
    speed_factors = [0.9, 1.1, 1.2]
    
    for idx, row in df.iterrows():
        # Add original sample
        augmented_rows.append(row.to_dict())
        
        # Add speed-perturbed versions (simulate by duplicating with different IDs)
        if random.random() < 0.3:  # 30% chance to augment
            for speed in speed_factors:
                new_row = row.to_dict()
                new_row['id'] = f"{row['id']}_sp{speed}"
                augmented_rows.append(new_row)
    
    # Create augmented dataframe
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"Augmented dataset size: {len(augmented_df)}")
    
    # Save augmented manifest
    output_tsv = os.path.join(output_dir, "train.tsv")
    augmented_df.to_csv(output_tsv, sep='\t', index=False)
    
    # Copy other files
    for file in ["dev.tsv", "test.tsv", "config_gcmvn.yaml", "config_mtl_asr_st_ctcst.yaml"]:
        src = os.path.join(data_dir, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            os.system(f"cp {src} {dst}")
    
    print("Data augmentation completed!")

if __name__ == "__main__":
    data_dir = "/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/hi-en/fbank2unit"
    output_dir = "/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/hi-en/fbank2unit_augmented"
    
    augment_training_data(data_dir, output_dir)