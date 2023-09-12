import argparse
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import train_test_split


def read_parquet_and_split(parquet_dir_paths, sample_size):
    ddfs = []
    for path in parquet_dir_paths:
        with ProgressBar():
            ddf = dd.read_parquet(path)
            ddfs.append(ddf)

    # Combine all the Dask DataFrames
    combined_ddf = dd.concat(ddfs)

    # Apply stratified sampling if sample_size is less than 1
    if sample_size < 1.0:
        total_length = len(combined_ddf)
        sample_length = int(total_length * sample_size)
        # Assuming the label column is named 'label'
        combined_ddf = combined_ddf.sample(frac=sample_size, random_state=42, replace=False).compute()
        combined_ddf = dd.from_pandas(combined_ddf, npartitions=combined_ddf.npartitions)

    # Split the data into training and test sets (Assuming the label column is named 'label')
    X_train, X_test = train_test_split(
        combined_ddf, test_size=0.2, shuffle=True, random_state=42, stratify=combined_ddf["label"]
    )

    return X_train, X_test


def list_of_strings(arg):
    return arg.split(",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split parquet data into train and test sets")
    parser.add_argument("-i", "--inputs", help="Parquet directories to read", required=True, type=list_of_strings)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-p", "--prefix", help="Prefix for the output files", required=True)
    parser.add_argument("-ss", "--sample-size", help="Sample size", required=False, type=float, default=1.0)

    args = parser.parse_args()
    parquet_dir_paths = args.inputs
    output_dir = args.output
    prefix = args.prefix
    sample_size = args.sample_size

    X_train, X_test = read_parquet_and_split(parquet_dir_paths, sample_size)

    with ProgressBar():
        X_train.repartition(npartitions=1).to_parquet(
            os.path.join(output_dir, f"{prefix}_train_data.parquet"),
        )
        X_test.repartition(npartitions=1).to_parquet(
            os.path.join(output_dir, f"{prefix}_test_data.parquet"),
        )

    print("Train and test data have been saved.")
