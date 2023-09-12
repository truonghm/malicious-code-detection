import argparse
import os
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import dask.dataframe as dd
import esprima
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from gensim.models import Word2Vec
from lib.utils.logging import logger
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


def list_of_strings(arg):
    return arg.split(",")


def tokenize(snippet: str) -> List[Tuple[Any, Any]]:
    token_types = []
    tokens = []
    try:
        result = esprima.tokenize(snippet)
        for token in result:
            # tokens.append((token.type, token.value))
            tokens.append(token.value)
            token_types.append(token.type)
    except Exception as e:
        # raise
        # tokens.append(("PARSE_ERROR", None))
        tokens.append("PARSE_ERROR")
        token_types.append("PARSE_ERROR")
    return tokens, token_types


def get_token(token_corpus, is_type: bool = False):
    ast = []
    for tokens in token_corpus:
        ast.append([token[0 if is_type else 1] for token in tokens])
    return ast


def train_word2vec(tokens, vector_size=100, window=5, min_count=1, workers=4):
    # tokens_by_type = get_token(corpus, True)
    # tokens_by_code = get_token(corpus, False)
    model = Word2Vec(sentences=tokens, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


def text_to_vector(model, text_tokens):
    vector = np.mean([model.wv[token] for token in text_tokens if token in model.wv.index_to_key], axis=0)
    return vector


def corpus_to_vectors(model, tokens, is_type=False):
    # tokens = get_token(corpus, is_type)
    return np.array([text_to_vector(model, text) for text in tokens])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform parquet data into word2vec vectors")
    parser.add_argument("-i", "--inputs", help="Parquet files to read", required=True, type=list_of_strings)
    parser.add_argument("-o", "--output", help="Output directory", required=True)

    args = parser.parse_args()
    parquet_file_paths = args.inputs
    output_dir = args.output

    for path in parquet_file_paths:
        # get file name using os.path
        logger.info(f"Processing {path}")
        file_name = os.path.basename(path)

        # with dd.read_parquet(path).compute() as df:
        with ProgressBar():
            if "test" in path:
                n = 100
            else:
                n = 1000
            df_good = dd.read_parquet(path, filters=[("label", "==", "good")]).head(int(n / 2), compute=True)
            df_bad = dd.read_parquet(path, filters=[("label", "==", "bad")]).tail(int(n / 2), compute=True)
            df = pd.concat([df_good, df_bad], axis=0)
            # df = dd.read_parquet(path).compute()

        # logger.info(df["label"].unique())

        logger.info("Tokenizing javascript snippets")
        token_results = [tokenize(content) for content in df["content"]]
        tokens, token_types = zip(*token_results)
        # logger.info(tokens[0])
        # logger.info(token_types[0])
        # break
        # df["tokens"] = pd.Series(tokens)
        # df["token_types"] = pd.Series(token_types)

        logger.info("Training word2vec model")
        model_by_val = train_word2vec(tokens, vector_size=100, window=5, min_count=1, workers=4)
        model_by_type = train_word2vec(token_types, vector_size=100, window=5, min_count=1, workers=4)

        # convert tokens to vectors
        logger.info("Converting tokens to vectors")
        type_vectors = corpus_to_vectors(model_by_type, token_types, True)
        val_vectors = corpus_to_vectors(model_by_val, tokens, False)
        # convert label to numpy array of 0 and 1
        label = np.array([1 if lb == "bad" else 0 for lb in df["label"]])

        # add label to vectors
        type_vectors = np.hstack((type_vectors, label.reshape(-1, 1)))
        val_vectors = np.hstack((val_vectors, label.reshape(-1, 1)))
        type_vectors_file_name = file_name.replace(".parquet", "_word2vec_by_type.parquet")
        val_vectors_file_name = file_name.replace(".parquet", "_word2vec_by_val.parquet")
        # Convert to Dask DataFrame and save to Parquet
        columns = ["c" + str(i) for i in range(100)] + ["label"]
        type_vectors_ddf = dd.from_pandas(pd.DataFrame(type_vectors, columns=columns), npartitions=5)
        val_vectors_ddf = dd.from_pandas(pd.DataFrame(val_vectors, columns=columns), npartitions=5)
        with ProgressBar():
            type_vectors_ddf.to_parquet(os.path.join(output_dir, type_vectors_file_name))
            val_vectors_ddf.to_parquet(os.path.join(output_dir, val_vectors_file_name))
