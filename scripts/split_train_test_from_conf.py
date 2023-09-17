import argparse
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

TRAIN_SET = "train_set.csv"
TEST_SET = "test_set.csv"
VAL_SET = "valid_set.csv"


def get_files_from_subdir(dir_path):
    return [
        os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, fname))
    ]


def traverse_path(path):
    good_files = []
    bad_files = []

    goodjs_dir = os.path.join(path, "goodjs")
    badjs_dir = os.path.join(path, "badjs")

    if not os.path.exists(goodjs_dir) and not os.path.exists(badjs_dir):
        print("No goodjs or badjs directory found in the path")

    if os.path.exists(goodjs_dir):
        goodjs_files = get_files_from_subdir(goodjs_dir)
        good_files.extend(goodjs_files)

    if os.path.exists(badjs_dir):
        badjs_files = get_files_from_subdir(badjs_dir)
        bad_files.extend(badjs_files)

    return good_files, bad_files


def sample_files(good_files, bad_files, good_sample_size, bad_sample_size):
    import random

    good_sample = random.sample(good_files, good_sample_size)
    bad_sample = random.sample(bad_files, bad_sample_size)

    return good_sample, bad_sample


def main():
    parser = argparse.ArgumentParser(description="Split text data into train and test sets")
    parser.add_argument("-c", "--config", help="Path to config", required=True, type=str)
    parser.add_argument(
        "-o", "--output", help="Output directory to contain the train and test set", required=True, type=str
    )
    parser.add_argument(
        "-ts", "--train-size", help="Size of the training set from sample (0.0 to 1.0)", type=float, default=0.8
    )

    args = parser.parse_args()

    path = args.config

    with open(path) as f:
        config = yaml.safe_load(f)
        input_conf = config["input"]

        output_dir = args.output

    total_good = 0
    total_bad = 0
    all_good_samples = []
    all_bad_samples = []
    for dir, conf in input_conf.items():
        dist_zero = {"dist": 0.0}
        dir_path = conf["path"]
        # print(dir_path)
        goodjs_dist = float(conf["classes"].get("goodjs", dist_zero)["dist"])
        badjs_dist = float(conf["classes"].get("badjs", dist_zero)["dist"])

        good_files, bad_files = traverse_path(dir_path)
        total_good += len(good_files)
        total_bad += len(bad_files)
        good_samples, bad_samples = sample_files(
            good_files, bad_files, int(goodjs_dist * len(good_files)), int(badjs_dist * len(bad_files))
        )
        all_good_samples.extend(good_samples)
        all_bad_samples.extend(bad_samples)

    print("Total good files : ", total_good)
    print("Total bad files  : ", total_bad)
    print("# of good samples: ", len(all_good_samples))
    print("# of bad samples : ", len(all_bad_samples))

    all_files = all_good_samples + all_bad_samples
    labels = ["goodjs"] * len(all_good_samples) + ["badjs"] * len(all_bad_samples)

    X_train, X_test, y_train, y_test = train_test_split(all_files, labels, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Train size       : ", len(X_train))
    print("Test size        : ", len(X_test))
    print("Val size         : ", len(X_val))

    train = pd.DataFrame({"file": X_train, "label": y_train})
    test = pd.DataFrame({"file": X_test, "label": y_test})
    val = pd.DataFrame({"file": X_val, "label": y_val})

    train_path = os.path.join(output_dir, TRAIN_SET)
    test_path = os.path.join(output_dir, TEST_SET)
    val_path = os.path.join(output_dir, VAL_SET)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    val.to_csv(val_path, index=False)
    print(f"Train set size    : {len(train)}")
    print(f"Test set size     : {len(test)}")
    print(f"Val set size      : {len(val)}")

    print(f"Output            : [{train_path}, {test_path}, {val_path}]")


if __name__ == "__main__":
    main()
