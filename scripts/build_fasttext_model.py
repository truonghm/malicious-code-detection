import argparse

import fasttext
from lib.utils.logging import logger

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, required=True)
    argparser.add_argument("-m", "--model-dir", type=str, default="models")
    argparser.add_argument("-hs", "--hierarchical-softmax", action=argparse.BooleanOptionalAction)

    args = argparser.parse_args()
    text_path = args.input
    model_path = args.model_dir
    hs = args.hierarchical_softmax

    logger.info("Training word vectors...")
    if hs:
        model = fasttext.train_unsupervised(text_path, model="skipgram", loss="hs")
        model.save_model(model_path)

    else:
        model = fasttext.train_unsupervised(text_path, model="skipgram")
        model.save_model(model_path)
