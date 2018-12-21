"""Build an np.array for a vocab file through random indexing.

The vocab file should be build e.g. by using Guillaume's `build_vocab.py`.

"""

__author__ = "Jussi Karlgren"

import sparsevectors
import numpy as np

dimensionality = 300
densedensity = 300
sparsedensity = 10

density = densedensity
labeldensity = sparsedensity

path = "/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/"
# read words file
if __name__ == '__main__':
    # Load vocab
    with open(path + "vocab.words.txt","r+") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    print("antal ord {}".format(size_vocab))

    # Array of zeros
    embeddings = np.zeros((size_vocab, dimensionality))

    for word in word_to_idx:
        vector = sparsevectors.newrandomvector(dimensionality, density)
        word_idx = word_to_idx[word]
        embeddings[word_idx] = sparsevectors.listify(vector, dimensionality)

    np.savez_compressed(path + 'randomindex.npz', embeddings=embeddings)

    with open(path + "vocab.words.txt","r+") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    print("antal ord {}".format(size_vocab))

    # Array of zeros
    embeddings = np.zeros((size_vocab, dimensionality))

    for word in word_to_idx:
        vector = sparsevectors.newrandomvector(dimensionality, density)
        word_idx = word_to_idx[word]
        embeddings[word_idx] = sparsevectors.listify(vector, dimensionality)

    np.savez_compressed(path + 'randomindex.npz', embeddings=embeddings)

    labels = {}
    for labelset in ["dependencies", "ner", "postags"]:
        with open(path + labelset + ".list","r+") as f:
            for line in f:
                labels[line.strip()] = sparsevectors.newrandomvector(dimensionality, labeldensity)

    print("antal etiketter {}".format(len(labels)))

    # Array of zeros
    embeddings = np.zeros((size_vocab, dimensionality))

    for word in word_to_idx:
        vector = sparsevectors.newrandomvector(dimensionality, density)
        word_idx = word_to_idx[word]
        embeddings[word_idx] = sparsevectors.listify(vector, dimensionality)

    np.savez_compressed(path + 'randomindexlabels.npz', embeddings=embeddings)

    

# 3.1 output
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)
# 3.2 output
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)

# generate permutations for each ner
# permute each index vector (try both dense and sparse cases)
# 4.1 output
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)
# 4.2 output
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)

# combine the two
# 34.1 output
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)
# 34.2 output

    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)

# generate permutation for each dep from a palette of interesting deps
# generate some operation to encode each triple from a similar palette

# generate index vector for each construction of interest

