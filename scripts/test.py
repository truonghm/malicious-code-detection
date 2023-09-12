import concurrent.futures
import os
import time

import esprima
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from tqdm import tqdm


def tokenize(snippet: str, label: str):
    token_types = []
    tokens = []
    try:
        result = esprima.tokenize(snippet)
        for token in result:
            tokens.append(token.value)
            token_types.append(token.type)
    except Exception as e:
        pass
    return tokens, token_types, label


def run(f, contents, labels, total):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(f, contents, labels), total=total))

        return results

def main():
    print("Loading data...")
    df = pd.read_parquet("data/exp/exk2_train_data.parquet/part.0.parquet", columns=["content", "label"])
    # pf = ParquetFile("data/exp/exk2_train_data.parquet/part.0.parquet")
    # first_ten_rows = next(pf.iter_batches(batch_size = 10))
    # df = pa.Table.from_batches([first_ten_rows]).to_pandas()

    print("Tokenizing...")
    results = run(tokenize, df["content"].to_list(), df["label"].to_list(), len(df))
    text_dir = "data/exp/"
    tokens_file = os.path.join(text_dir, "tokens_corpus.txt")
    token_types_file = os.path.join(text_dir, "token_types_corpus.txt")
    labels_file = os.path.join(text_dir, "labels.txt")

    print("Saving corpus to text files...")
    with open(tokens_file, "a") as tokens_f, open(token_types_file, "a") as token_types_f, open(
        labels_file, "a"
    ) as labels_f:
        for tokens, token_types, label in tqdm(results, total=len(results)):
            tokens_f.write(" ".join(tokens) + "\n")
            token_types_f.write(" ".join(token_types) + "\n")
            labels_f.write(label + "\n")

if __name__ == "__main__":
    main()
