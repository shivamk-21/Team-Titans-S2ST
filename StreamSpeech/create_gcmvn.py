#!/usr/bin/env python3
"""
Create GCMVN (Global Cepstral Mean and Variance Normalization) statistics
"""

import numpy as np
import zipfile
import os
from pathlib import Path
import argparse

def extract_fbank_features_from_zip(zip_path, max_files=1000):
    """Extract a sample of fbank features from the zip file"""
    features_list = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_list = zip_file.namelist()
        npy_files = [f for f in file_list if f.endswith('.npy')]
        
        # Sample a subset of files
        sample_files = npy_files[:min(max_files, len(npy_files))]
        
        print(f"Processing {len(sample_files)} files for GCMVN statistics...")
        
        for file_name in sample_files:
            try:
                with zip_file.open(file_name) as f:
                    features = np.load(f)
                    if len(features.shape) == 2:  # Should be (time, features)
                        features_list.append(features)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
    
    return features_list

def compute_gcmvn_stats(features_list):
    """Compute global mean and variance from features"""
    if not features_list:
        # Create dummy statistics if no features available
        print("Warning: No features found, creating dummy GCMVN statistics")
        return {
            'mean': np.zeros(80),  # Assuming 80-dim fbank features
            'std': np.ones(80)
        }
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    # Compute mean and std
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    # Prevent division by zero
    std = np.where(std == 0, 1.0, std)
    
    print(f"Computed GCMVN stats from {all_features.shape[0]} frames")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"Std range: [{std.min():.3f}, {std.max():.3f}]")
    
    return {
        'mean': mean,
        'std': std
    }

def create_gcmvn_file(lang):
    """Create GCMVN file for a language"""
    
    # Path to the source fbank zip file
    zip_path = Path(f"/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets/cvss/cvss-c/{lang}-en/src_fbank80.zip")
    
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found, creating dummy GCMVN stats")
        features_list = []
    else:
        features_list = extract_fbank_features_from_zip(zip_path)
    
    # Compute statistics
    stats = compute_gcmvn_stats(features_list)
    
    # Save to configs directory
    output_path = Path(f"/run/media/shivamk21/data/ML-Project/StreamSpeech/configs/{lang}-en/gcmvn.npz")
    output_path.parent.mkdir(exist_ok=True)
    
    np.savez(output_path, mean=stats['mean'], std=stats['std'])
    print(f"Created GCMVN file: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code (hi, ma, mr)")
    args = parser.parse_args()
    
    create_gcmvn_file(args.lang)

if __name__ == "__main__":
    main()