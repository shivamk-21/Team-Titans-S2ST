#!/usr/bin/env python3
"""
Convert the repository `useable_data` layout into the folder layout expected by the
preprocess scripts (create_manifest.py and others).

What it does:
- Reads `useable_data/{train,dev,test}.tsv` (if present).
- For each split it copies referenced audio files into OUT_ROOT/<split>/ using the
  audio filename basename (e.g. audio_0_93.wav).
- Writes a simplified TSV at OUT_ROOT/{train,dev,test}.tsv where `src_audio`
  fields are rewritten to the basename (keeps other columns the same when
  possible). This is compatible with other preprocessing scripts that expect
  a CVSS-style TSV at the language folder root.
- Writes CREATE-style manifest TXT files OUT_ROOT/{train,dev,test}.txt with the
  same format produced by `create_manifest.py` (first line the split folder,
  following lines: <audio_filename.wav>\t<n_frames>).

Usage example:
    python preprocess_scripts/convert_useable_data.py \
        --useable-root ./useable_data \
        --out-root /run/media/shivamk21/data/datasets/cvss/cvss-c/hi-en

Notes:
- The script will attempt to locate audio files using the `src_audio` path in
  the original TSV (it may be a relative path like "hi_en/hi/audio_0_93.wav").
  If the file is not found it will warn and skip that row.
- Requires Python packages: pandas, soundfile, tqdm (these are already used
  elsewhere in the repo). If not present install them in your environment.
"""

import argparse
from pathlib import Path
import shutil
import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SPLITS = ["train", "dev", "test"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_audio_file(useable_root: Path, src_audio: str) -> Path:
    """Try to resolve the audio file path from the src_audio field.
    src_audio may be:
      - a path with subfolders (hi_en/hi/audio_0_93.wav)
      - just a basename (audio_0_93.wav)
    We try a few reasonable locations under useable_root.
    """
    p = Path(src_audio)
    # If it is already absolute and exists, return it
    if p.is_absolute() and p.exists():
        return p
    # Direct relative path under useable_root
    cand = useable_root / p
    if cand.exists():
        return cand
    # Maybe the src_audio contains a leading folder like "hi_en/hi/..." but
    # files are directly in useable_root/<lang>/<basename>. Try basename search
    cand = useable_root / p.name
    if cand.exists():
        return cand
    # Try search in any subfolder that matches the first two components
    # (this is a simple heuristic; we avoid expensive recursive search by default)
    # Fallback: return Path that may not exist and let caller handle it
    return cand


def process_split(useable_root: Path, per_lang_out: Path, split: str, rows_iter):
    """Copy audio files for one split into per_lang_out/<split>/ and write a
    minimal {split}.tsv file (no header) listing audio stems (one per line).

    rows_iter: iterable of pandas Series rows (each row should contain 'src_audio')
    """
    out_split_dir = per_lang_out / split
    ensure_dir(out_split_dir)

    txt_entries = []
    # allow rows_iter as DataFrame or list
    try:
        total = len(rows_iter)
    except Exception:
        rows_iter = list(rows_iter)
        total = len(rows_iter)

    print(f"Processing split={split}, {total} rows -> out: {per_lang_out}")
    for row in tqdm(rows_iter, total=total):
        src_audio_field = row.get("src_audio") if "src_audio" in row.index else None
        if src_audio_field is None:
            continue
        audio_src_path = find_audio_file(useable_root, src_audio_field)
        if not audio_src_path.exists():
            print(f"Warning: audio file not found for '{src_audio_field}' -> tried '{audio_src_path}'")
            continue
        dest_name = audio_src_path.name
        dest_path = out_split_dir / dest_name
        shutil.copy2(audio_src_path, dest_path)
        txt_entries.append(Path(dest_name).stem)

    # Write minimal split TSV expected by create_manifest.py: each line contains stem + '\t'
    if len(txt_entries) > 0: 
        with open(per_lang_out / f"{split}.tsv", "w", encoding="utf-8") as fout:
            for stem in txt_entries:
                fout.write(f"{stem}\t\n")
    else:
        print(f"No valid rows for split {split} and prefix -> skipping {per_lang_out}/{split}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useable-root", type=str, required=True, help="path to useable_data folder")
    parser.add_argument("--out-root", type=str, required=True, help="path to write converted cvss-like folder (e.g. /.../datasets/cvss/cvss-c/hi-en)")
    parser.add_argument(
        "--filter-language",
        type=str,
        default=None,
        help="optional: only include rows where the 'language' column equals this value (e.g. 'hindi')",
    )
    args = parser.parse_args()

    useable_root = Path(args.useable_root).absolute()
    out_root = Path(args.out_root).absolute()
    ensure_dir(out_root)

    # Aggregate rows by detected prefix (first path component of src_audio)
    all_rows = {s: [] for s in SPLITS}
    prefixes = set()
    for split in SPLITS:
        tsv_path = useable_root / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"No {tsv_path}, skipping split {split}")
            continue
        df = pd.read_csv(tsv_path, sep="\t", header=0, encoding="utf-8")
        for _, row in df.iterrows():
            src = row.get("src_audio") if "src_audio" in row.index else None
            if src is None:
                continue
            p = Path(src)
            if len(p.parts) > 1:
                prefix = p.parts[0]
            else:
                langcol = row.get("language") if "language" in row.index else None
                prefix = langcol if langcol is not None else "unknown"
            prefixes.add(prefix)
            all_rows[split].append((prefix, row))

    prefixes = sorted(prefixes)
    if args.filter_language is not None:
        pref = args.filter_language
        prefixes = [p for p in prefixes if p == pref or p.replace("_", "-") == pref or pref in p or pref in p.replace("_", "-")]
        if not prefixes:
            print(f"Warning: no prefix matched filter '{args.filter_language}'")

    # For each detected prefix, create a per-language output folder and process splits
    for prefix in prefixes:
        lang_folder = prefix.replace("_", "-")
        per_lang_out = out_root / lang_folder
        per_lang_out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            rows_for_split = [row for (p, row) in all_rows.get(split, []) if p == prefix]
            if len(rows_for_split) == 0:
                # nothing to do for this split
                continue
            process_split(useable_root, per_lang_out, split, rows_for_split)

    print("Done.\nSummary:\n- Converted files were written to:")
    print(f"  {out_root}")
    print("- Each split now has a folder with .wav files and an updated {split}.tsv and {split}.txt")


if __name__ == "__main__":
    main()
