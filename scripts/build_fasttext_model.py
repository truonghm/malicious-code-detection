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
        model = fasttext.train_unsupervised(text_path, model="cbow", loss="hs")
        model.save_model(model_path)

    else:
        model = fasttext.train_unsupervised(text_path, model="cbow")
        model.save_model(model_path)

    vec_path = model_path.replace(".bin", "") + ".vec"
    # model.wv.save_word2vec_format(vec_path)
    words = model.get_words()
    with open(vec_path, "w") as vec_out:
        vec_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

        for w in words:
            v = model.get_word_vector(w)
            v_str = ""
            for vi in v:
                v_str += " " + str(vi)

            try:
                vec_out.write(w + v_str + "\n")
            except Exception as e:
                logger.error(e)
                continue
