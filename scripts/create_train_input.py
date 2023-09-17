import numpy as np
import pandas as pd
import pyarrow.parquet as pq


print("Convert paths to one single parquet file")
TRAIN_PATH = "data/exp/train_set.csv"
TEST_PATH = "data/exp/test_set.csv"
VALID_PATH = "data/exp/valid_set.csv"

train_sources = pd.read_csv(TRAIN_PATH)
test_sources = pd.read_csv(TEST_PATH)
valid_sources = pd.read_csv(VALID_PATH)


def get_content(path):
    with open(path, "r") as f:
        return f.read().strip()


def convert(sources, format, path):
    sources["code"] = sources["file"].apply(get_content)

    sources.drop(columns=["file"], inplace=True)

    sources["label"] = np.where(sources["label"] == "goodjs", 0, 1)
    sources_name = f'{sources=}'.split('=')[0]
    print(f"length of {sources_name}:", len(sources))

    if format == "parquet":
        parquet_path = path.replace(".csv", "") + ".parquet"
        train_sources.to_parquet(parquet_path, engine="pyarrow", index=False)
    elif format == "jsonl":
        jsonl_path = path.replace(".csv", "") + ".jsonl"
        with open(jsonl_path, "w") as f:
            f.write(sources.to_json(orient='records', lines=True, force_ascii=False))


for df, path in zip(
    [train_sources, test_sources, valid_sources],
    [TRAIN_PATH, TEST_PATH, VALID_PATH],
):
    # convert(df, "parquet", path)
    convert(df, "jsonl", path)
