import fasttext
import numpy as np

# Load the trained FastText model
model = fasttext.load_model("models/fasttext_model_by_types.bin")
with open("data/exp/token_types_corpus.txt", "r") as tokens_f, open("data/exp/labels.txt", "r") as labels_f:
    token_corpus = tokens_f.readlines()
    labels = labels_f.readlines()


def text_to_vector(tokens):
    vectors = [model.get_word_vector(token) for token in tokens]

    if len(vectors) == 0:
        return np.zeros(model.get_dimension())

    vectors = np.array(vectors)

    # Aggregate the vectors (e.g., by averaging them)
    aggregated_vector = np.mean(vectors, axis=0)

    return aggregated_vector

def corpus_to_vector(corpus):
    return np.array([text_to_vector(text) for text in corpus])


token_vectors = corpus_to_vector(token_corpus)
label_vectors = corpus_to_vector(labels)

# Save the vectors
np.save("data/exp/token_vectors.npy", token_vectors)
np.save("data/exp/label_vectors.npy", label_vectors)
