import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SET = "train_set.csv"
TEST_SET = "test_set.csv"


def list_of_strings(arg):
    return arg.split(",")


def get_files_from_subdir(dir_path):
    return [
        os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, fname))
    ]


def main():
    parser = argparse.ArgumentParser(description="Split text data into train and test sets")
    parser.add_argument("-i", "--inputs", help="Directories to read", required=True, type=list_of_strings)
    parser.add_argument(
        "-o", "--output", help="Output directory to contain the train and test set", required=True, type=str
    )
    parser.add_argument("-ss", "--sample-size", help="Size of the sample to use (0.0 to 1.0)", type=float, default=0.2)
    parser.add_argument(
        "-ts", "--train-size", help="Size of the training set from sample (0.0 to 1.0)", type=float, default=0.8
    )

    args = parser.parse_args()

    print(f"Inputs            : {args.inputs}")
    print(f"Sample size       : {args.sample_size*100} %")
    print(f"Train size        : {args.train_size*100} %")
    print(f"Test size         : {round(1.0 - args.train_size, 1)*100} %")

    good_files = []
    bad_files = []
    labels = []

    for input_dir in args.inputs:
        goodjs_dir = os.path.join(input_dir, "goodjs")
        badjs_dir = os.path.join(input_dir, "badjs")

        if os.path.exists(goodjs_dir) and os.path.exists(badjs_dir):
            goodjs_files = get_files_from_subdir(goodjs_dir)
            badjs_files = get_files_from_subdir(badjs_dir)

            good_files.extend(goodjs_files)
            bad_files.extend(badjs_files)

            labels.extend(["goodjs"] * len(goodjs_files))
            labels.extend(["badjs"] * len(badjs_files))
        else:
            print(f"Skipping {input_dir} as it doesn't contain both 'goodjs' and 'badjs' directories.")

    all_files = good_files + bad_files
    total = len(all_files)
    print(f"# before sampling : {total}")
    print(f"# goodjs          : {len(good_files)}")
    print(f"# badjs           : {len(bad_files)}")
    # Sample from the data if necessary
    if args.sample_size < 1.0:
        sample_size = int(len(all_files) * args.sample_size)
        all_files, _, labels, _ = train_test_split(
            all_files, labels, train_size=sample_size, stratify=labels, random_state=42
        )

    good_files_after_sample = [all_files[i] for i in range(len(all_files)) if labels[i] == "goodjs"]
    bad_files_after_sample = [all_files[i] for i in range(len(all_files)) if labels[i] == "badjs"]
    print(f"# after sampling  : {len(all_files)}")
    print(f"# goodjs sampled  : {len(good_files_after_sample)}")
    print(f"# badjs sampled   : {len(bad_files_after_sample)}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, labels, train_size=args.train_size, stratify=labels, random_state=42
    )

    output_dir = args.output

    train = pd.DataFrame({"file": X_train, "label": y_train})
    test = pd.DataFrame({"file": X_test, "label": y_test})
    train_path = os.path.join(output_dir, TRAIN_SET)
    test_path = os.path.join(output_dir, TEST_SET)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Train set size    : {len(train)}")
    print(f"Test set size     : {len(test)}")
    print(f"Output            : [{train_path}, {test_path}]")


if __name__ == "__main__":
    main()
