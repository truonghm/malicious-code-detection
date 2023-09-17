import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import esprima
import pandas as pd
from lib.utils.logging import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")


def tokenize(snippet: str) -> List[Tuple[Any, Any]]:
    token_types = []
    tokens = []
    try:
        result = esprima.tokenize(snippet)
        for token in result:
            tokens.append(token.value)
            token_types.append(token.type)
    except Exception as e:
        pass
    return token_types


def process_row(file, label):
    with open(file, "r") as f:
        snippet = f.read()
    return tokenize(snippet), label


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, required=True)
    argparser.add_argument("-o", "--output", type=str, required=True)

    args = argparser.parse_args()
    path_file = args.input
    text_dir = args.output
    prefix = os.path.basename(path_file).replace(".csv", "")
    token_types_file = os.path.join(text_dir, f"{prefix}_token_types_corpus.txt")
    labels_file = os.path.join(text_dir, f"{prefix}_labels.txt")

    logger.info(f"Load paths to actual data: {path_file}")
    df = pd.read_csv(path_file)
    total = len(df)
    logger.info(f"Total number of files: {total}")

    # logger.info("Tokenizing...")
    results = []
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_row, df["file"].tolist(), df["label"].tolist()),
                total=total,
                bar_format="Tokenizing: {desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}",
            )
        )

    logger.info(f"Number of observation tokenized: {len(results)}")
    logger.info("Saving corpus to text files...")
    with open(token_types_file, "a") as token_types_f, open(
        labels_file, "a"
    ) as labels_f:
        for token_types, label in results:
            token_types_f.write(" ".join(token_types) + "\n")
            labels_f.write(label + "\n")


if __name__ == "__main__":
    main()
