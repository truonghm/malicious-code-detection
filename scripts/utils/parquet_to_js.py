import argparse
import hashlib
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm


def create_text_files(parquet_path, root_dir):
    with ProgressBar():
        # Read the parquet file
        ddf = dd.read_parquet(parquet_path)

        # Compute to bring into memory (use this cautiously)
        df = ddf.compute()

    # Create root directory named after the parquet file
    root_path = os.path.join(root_dir, os.path.basename(parquet_path).replace(".parquet", ""))
    os.makedirs(root_path, exist_ok=True)

    # Create subdirectories
    goodjs_path = os.path.join(root_path, "goodjs")
    badjs_path = os.path.join(root_path, "badjs")
    os.makedirs(goodjs_path, exist_ok=True)
    os.makedirs(badjs_path, exist_ok=True)

    # Create text files
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row["label"]
        content = row["content"]
        hash_value = hashlib.sha256(content.encode()).hexdigest()

        if label == "good":
            file_path = os.path.join(goodjs_path, hash_value)
        else:
            file_path = os.path.join(badjs_path, hash_value)

        with open(file_path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create text files from a parquet file")
    parser.add_argument("-i", "--input", help="Input parquet file path", required=True)
    parser.add_argument("-o", "--output", help="Output root directory path", required=True)

    args = parser.parse_args()
    parquet_path = args.input
    root_dir = args.output

    create_text_files(parquet_path, root_dir)
