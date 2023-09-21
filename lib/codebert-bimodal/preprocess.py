import random

import pandas as pd

random.seed(123456)

def create_doc(js):
    malicious_doc = "javascript perform malicious actions to trick users, steal data from users, \
        or otherwise cause harm."
    benign_doc = "javascript perform normal, non-harmful actions"

    label = js["label"]
        # choose randomly between malicious and benign doc
    if random.random() < 0.5:
        doc = malicious_doc
        new_label = 1 if label == 1 else 0
    else:
        doc = benign_doc
        new_label = 1 if label == 0 else 0

    # js["label"] = new_label
    # js["doc"] = doc

    # return js
    return doc, new_label

def modify_dataset(file_path, type:str):
    with open (file_path, "r") as f:
        # convert jsonl file to pandas
        df = pd.read_json(f, lines=True)

    df[["doc", "label"]] = df.apply(create_doc, axis=1, result_type="expand")
    df["idx"] = type + "_" + df.index.astype(str)
    new_path = file_path.replace(".jsonl", "_new.jsonl")
    with open(new_path, "w") as f:
        f.write(df.to_json(orient='records', lines=True, force_ascii=False))


if __name__ == "__main__":
    print("modifying test set")
    modify_dataset("data/exp/test_set.jsonl", "test")
    print("modifying valid set")
    modify_dataset("data/exp/valid_set.jsonl", "valid")
    print("modifying train set")
    modify_dataset("data/exp/train_set.jsonl", "train")
