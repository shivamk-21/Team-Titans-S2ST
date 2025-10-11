import os
import pandas as pd
import shutil
from pathlib import Path
import wave
from tqdm import tqdm

def get_audio_duration_frames(audio_path):
    """Get audio duration in frames (samples)"""
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            return wav_file.getnframes()
    except Exception as e:
        # If wave doesn't work, try a simple estimation based on file size
        try:
            file_size = os.path.getsize(audio_path)
            # Rough estimation: assuming 16kHz, 16-bit audio
            estimated_frames = file_size // 2  # 2 bytes per sample for 16-bit
            return estimated_frames
        except:
            print(f"Error reading {audio_path}: {e}")
            return 0

def create_language_mapping():
    """Create mapping from language codes to language names"""
    return {
        'hi': 'hindi',
        'ma': 'malayalam', 
        'mr': 'marathi'
    }

def process_useable_data_to_datasets(useable_data_root, output_root):
    """
    Process useable_data format into CoVoST2 and CVSS-C style datasets
    """
    useable_data_path = Path(useable_data_root)
    output_path = Path(output_root)
    
    # Language mapping
    lang_mapping = create_language_mapping()
    
    # Read the main TSV files
    for split in ['train', 'dev', 'test']:
        split_file = useable_data_path / f"{split}.tsv"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping {split}")
            continue
            
        print(f"Processing {split} split...")
        df = pd.read_csv(split_file, sep='\t')
        df = df.iloc[:50000] if split == 'train' else df.iloc[:11000]  
        # Group by language pairs
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            src_audio = row['src_audio']
            tgt_audio = row['tgt_audio']
            language = row['language']
            src_text = row['src_text']
            tgt_text = row['tgt_text']
            
            # Extract language pair from audio path (e.g., 'hi_en/hi/audio_0_93.wav' -> 'hi-en')
            lang_pair = src_audio.split('/')[0].replace('_', '-')
            src_lang = lang_pair.split('-')[0]
            tgt_lang = lang_pair.split('-')[1]
            
            # Create directory structure for this language pair
            create_covost2_structure(output_path, src_lang, tgt_lang, split, row, useable_data_path)
            create_cvss_c_structure(output_path, src_lang, tgt_lang, split, row, useable_data_path)

def create_covost2_structure(output_path, src_lang, tgt_lang, split, row, useable_data_path):
    """Create CoVoST2 style structure"""
    covost_root = output_path / "covost2" / src_lang
    covost_root.mkdir(parents=True, exist_ok=True)
    
    # Create clips directory
    clips_dir = covost_root / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    # Copy source audio file
    src_audio_path = useable_data_path / row['src_audio']
    tgt_audio_path = useable_data_path / row['tgt_audio']
    audio_filename = os.path.basename(row['src_audio'])
    
    # Copy audio file to clips directory
    if src_audio_path.exists():
        shutil.copy2(src_audio_path, clips_dir / audio_filename)
    
    # Create or append to covost TSV file
    covost_tsv = covost_root / f"covost_v2.{src_lang}_{tgt_lang}.tsv"
    
    # Check if file exists to write header
    write_header = not covost_tsv.exists()
    
    with open(covost_tsv, 'a', encoding='utf-8') as f:
        if write_header:
            f.write("path\ttranslation\tsplit\n")
        f.write(f"{audio_filename}\t{row['tgt_text']}\t{split}\n")
    
    # Create validated.tsv (simplified version - in real CoVoST this has more fields)
    validated_tsv = covost_root / "validated.tsv"
    write_validated_header = not validated_tsv.exists()
    
    with open(validated_tsv, 'a', encoding='utf-8') as f:
        if write_validated_header:
            f.write("path\tsentence\tsplit\n")
        f.write(f"{audio_filename}\t{row['src_text']}\t{split}\n")

def create_cvss_c_structure(output_path, src_lang, tgt_lang, split, row, useable_data_path):
    """Create CVSS-C style structure"""
    lang_pair = f"{src_lang}-{tgt_lang}"
    cvss_root = output_path / "cvss" / "cvss-c" / lang_pair
    cvss_root.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    split_dir = cvss_root / split
    split_dir.mkdir(exist_ok=True)
    
    # Copy source and target audio files
    src_audio_path = useable_data_path / row['src_audio']
    tgt_audio_path = useable_data_path / row['tgt_audio']
    audio_filename = os.path.basename(row['src_audio'])
    tgt_audio_filename = os.path.basename(row['tgt_audio'])
    
    if src_audio_path.exists():
        shutil.copy2(src_audio_path, split_dir / audio_filename)
    if tgt_audio_path.exists():
        shutil.copy2(tgt_audio_path, split_dir / tgt_audio_filename)
    
    # Create/append to TSV file (just audio IDs without extension)
    tsv_file = cvss_root / f"{split}.tsv"
    audio_id = os.path.splitext(audio_filename)[0]
    
    with open(tsv_file, 'a', encoding='utf-8') as f:
        f.write(f"{audio_id}\t\n")
    
    # Create/append to txt file (audio path and frame count)
    txt_file = cvss_root / f"{split}.txt"
    write_txt_header = not txt_file.exists()
    
    # Get audio duration in frames
    audio_frames = get_audio_duration_frames(src_audio_path)
    
    with open(txt_file, 'a', encoding='utf-8') as f:
        if write_txt_header:
            f.write(f"{split_dir}\n")
        f.write(f"{audio_filename}\t{audio_frames}\n")
    
    # Create short TSV file (for reference texts)
    short_tsv = cvss_root / f"{split}.short.tsv"
    write_short_header = not short_tsv.exists()
    
    with open(short_tsv, 'a', encoding='utf-8') as f:
        if write_short_header:
            f.write("audio_id\tsrc_text\ttgt_text\n")
        f.write(f"{audio_id}\t{row['src_text']}\t{row['tgt_text']}\n")

def main():
    # Paths
    useable_data_root = "/run/media/shivamk21/data/ML-Project/Datasets/original"
    output_root = "/run/media/shivamk21/data/ML-Project/Datasets/processed_datasets"
    
    print("Starting dataset conversion...")
    print(f"Input: {useable_data_root}")
    print(f"Output: {output_root}")
    
    # Process the data
    process_useable_data_to_datasets(useable_data_root, output_root)
    
    print("Dataset conversion completed!")
    print(f"CoVoST2 style datasets created in: {output_root}/covost2/")
    print(f"CVSS-C style datasets created in: {output_root}/cvss/cvss-c/")

if __name__ == "__main__":
    main()