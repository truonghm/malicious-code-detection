import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import esprima
import numpy as np
import pandas as pd
from src.utils.logging import logger
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


def process_row(code, pbar):
    tokens = tokenize(code)
    pbar.update(1)
    return tokens


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, required=True)
    argparser.add_argument("-o", "--output", type=str, required=True)

    args = argparser.parse_args()
    path_file = args.input
    text_dir = args.output
    prefix = os.path.basename(path_file).replace(".jsonl", "")
    token_types_file = os.path.join(text_dir, f"{prefix}_token_types_corpus.parquet")

    logger.info(f"Load paths to actual data: {path_file}")
    with open(path_file, "r") as f:
        df = pd.read_json(f, lines=True)
    total = len(df)
    logger.info(f"Total number of files: {total}")

    # logger.info("Tokenizing...")
    bar_format = "Tokenizing: {desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    with tqdm(total=total, bar_format=bar_format) as pbar:
        df["ast"] = np.vectorize(process_row)(df["code"], pbar)

    logger.info("Saving corpus to text files...")

    df[["ast", "label"]].to_parquet(token_types_file, index=False)


if __name__ == "__main__":
    main()
