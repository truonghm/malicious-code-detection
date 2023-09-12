import argparse
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import train_test_split


def read_parquet_and_split(parquet_dir_paths):
    """
    Reads multiple parquet directories into a Dask DataFrame and splits it into train and test sets.

    Parameters:
    - parquet_dir_paths: list of str, paths to the parquet directories

    Returns:
    - X_train, X_test: Dask DataFrames, the training and test sets
    """
    ddfs = []
    for path in parquet_dir_paths:
        with ProgressBar():
            ddf = dd.read_parquet(path)
            ddfs.append(ddf)

    # Combine all the Dask DataFrames
    combined_ddf = dd.concat(ddfs)

    # Split the data into training and test sets
    X_train, X_test = train_test_split(combined_ddf, test_size=0.2, shuffle=True, random_state=42)

    return X_train, X_test


def list_of_strings(arg):
    return arg.split(",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split parquet data into train and test sets")
    parser.add_argument("-i", "--inputs", help="Parquet directories to read", required=True, type=list_of_strings)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-p", "--prefix", help="Prefix for the output files", required=True)
    args = parser.parse_args()
    parquet_dir_paths = args.inputs
    output_dir = args.output
    prefix = args.prefix

    X_train, X_test = read_parquet_and_split(parquet_dir_paths)

    with ProgressBar():
        X_train.repartition(npartitions=1).to_parquet(
            os.path.join(output_dir, f"{prefix}_train_data.parquet"),
        )
        X_test.repartition(npartitions=1).to_parquet(
            os.path.join(output_dir, f"{prefix}_test_data.parquet"),
        )

    print("Train and test data have been saved.")
