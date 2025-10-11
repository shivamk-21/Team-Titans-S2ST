import os
import argparse
import sys
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from examples.speech_to_text.data_utils import load_df_from_tsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--wav-list", type=str, required=True)
    parser.add_argument("--output-src", type=str, required=True)
    args = parser.parse_args()
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())

    d = {}
    for item in data:
        d[item["id"]] = item["src_text"]
    with open(args.wav_list, "r") as f_wav:
        wav = f_wav.read().splitlines()
    with open(args.output_src, "w") as f_src:
        for x in wav:
            # Extract audio ID from the path (last part after /)
            audio_id = x.split("/")[-1]
            if audio_id in d:
                src = d[audio_id]
                src = re.sub(r"[^\w\s]", "", src.lower())
                f_src.write(src + "\n")
            else:
                print(f"Warning: ID {audio_id} not found in TSV data")


if __name__ == "__main__":
    main()
