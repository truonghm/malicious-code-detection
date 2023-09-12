import argparse
import os
from typing import Any, List, Tuple

import dask.dataframe as dd
import esprima
from dask.diagnostics import ProgressBar
from lib.utils.logging import logger
from tqdm import tqdm


def tokenize(snippet: str, label) -> List[Tuple[Any, Any]]:
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


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--parquet-dir", type=str, required=True)
    argparser.add_argument("-st", "--text-dir", type=str, required=True)

    args = argparser.parse_args()
    path = args.parquet_dir
    text_dir = args.text_dir
    tokens_file = os.path.join(text_dir, "tokens_corpus.txt")
    token_types_file = os.path.join(text_dir, "token_types_corpus.txt")
    labels_file = os.path.join(text_dir, "labels.txt")

    logger.info("Reading parquet files...")
    with ProgressBar():
        ddf = dd.read_parquet(path)

    logger.info("Saving corpus to text files...")
    total = len(ddf)
    logger.info(f"Total snippets: {total}")

    df = ddf.compute()
    total = len(df)
    # with tqdm(total=total) as pbar:
    with open(tokens_file, "a") as tokens_f, open(token_types_file, "a") as token_types_f, open(
    labels_file, "a") as labels_f:
        for snippet, label in tqdm(zip(df["content"], df["label"]), total=total):
            tokens, token_types, label = tokenize(snippet, label)
            tokens_f.write(" ".join(tokens) + "\n")
            token_types_f.write(" ".join(token_types) + "\n")
            labels_f.write(label + "\n")
                # pbar.update(1)

if __name__ == "__main__":
    main()
