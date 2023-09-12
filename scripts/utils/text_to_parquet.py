import argparse
import os

import dask.bag as db
import dask.dataframe as dd
import pyarrow as pa
from dask.diagnostics import ProgressBar


def list_of_strings(arg):
    return arg.split(",")


# Function to process each text file and return a dictionary
def process_file(file_path, label):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {"js_len": len(content), "content": content, "label": label}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text files to Parquet")

    parser.add_argument("-g", "--good", help="Directory containing files labeled as good", type=list_of_strings)
    parser.add_argument("-b", "--bad", help="Directory containing files labeled as bad", type=list_of_strings)
    parser.add_argument("-p", "--parquet", help="Parquet file to output", required=True, type=str)
    args = parser.parse_args()

    good_dirs = args.good
    bad_dirs = args.bad
    if not good_dirs and not bad_dirs:
        raise ValueError("At least one directory must be specified")

    parquet_file_path = args.parquet

    good_files = []
    if good_dirs:
        for d in good_dirs:
            good_files.extend(
                [(os.path.join(d, f), "good") for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            )
    print(f"Found {len(good_files)} good files")

    bad_files = []
    if bad_dirs:
        for d in bad_dirs:
            bad_files.extend([(os.path.join(d, f), "bad") for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    print(f"Found {len(bad_files)} bad files")

    all_files = good_files + bad_files
    print(f"Found {len(all_files)} files in total")

    bag = db.from_sequence(all_files).map(lambda x: process_file(*x))

    ddf = bag.to_dataframe()

    schema = pa.schema([("js_len", pa.int64()), ("content", pa.string()), ("label", pa.string())])

    append_data = os.path.exists(parquet_file_path)

    with ProgressBar():
        ddf.to_parquet(
            parquet_file_path,
            schema=schema,
            append=append_data,
        )

    print(f"Data has been written to {parquet_file_path}")
