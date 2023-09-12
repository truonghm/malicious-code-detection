import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV to Parquet')
    parser.add_argument('-c', '--csv', help='CSV file to convert', required=True, type=str)
    parser.add_argument('-p', '--parquet', help='Parquet file to output', required=True, type=str)
    parser.add_argument('-cl', '--cols', help='List of columns to include in the Parquet file', type=list_of_ints)
    args = parser.parse_args()

    csv_file = args.csv
    parquet_file = args.parquet
    # get the column list, if needed convert it to a list
    column_list = args.cols

    # column_list should be a list of integers
    if column_list:
        try:
            column_list = [int(x) for x in column_list]
        except ValueError:
            raise ValueError("column_list must be a list of integers")

    csv_chunksize = 50000  # You can adjust this number based on your available memory and file row size

    # Calculate the total number of chunks that will be read for progress tracking
    total_rows = sum(1 for row in open(csv_file, 'r')) - 1  # Subtract 1 for the header
    num_chunks = (total_rows // csv_chunksize) + 1

    # Infer schema for our resulting parquet file (can be skipped if you want to specify a schema manually)
    if not column_list:
        first_chunk = pd.read_csv(csv_file, nrows=5)
    else:
        first_chunk = pd.read_csv(csv_file, nrows=5, usecols=column_list)
    schema = pa.Schema.from_pandas(df=first_chunk)

    # Initialize Parquet writer
    writer = None

    kwargs = {
        "chunksize": csv_chunksize,
    }

    if column_list:
        kwargs["usecols"] = column_list

    for chunk in tqdm(pd.read_csv(csv_file, **kwargs), total=num_chunks):
        table = pa.Table.from_pandas(chunk, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, table.schema)
        writer.write_table(table=table)

    # Finalize the writes and close
    if writer:
        writer.close()
