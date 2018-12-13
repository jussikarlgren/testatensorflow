"""Build an np.array for a vocab file through random indexing.

The vocab file should be build e.g. by using Guillaume's `build_vocab.py`.

"""


import sparsevectors
import numpy as np
from hyperdimensionalsemanticspace import SemanticSpace
from pathlib import Path


dimensionality = 300
dense = 150
sparse = 10

__author__ = "Jussi Karlgren"


# read words file
if __name__ == '__main__':
    # Load vocab
    with Path('vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, dimensionality))

 # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path('glove.840B.300d.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

# read pos file
# read ner file


# generate index vector for each token
dense = False

if dense:
    semanticspace = SemanticSpace(300, 150)
    # 1 output in npz form
    # Save np.array to file
    #    np.savez_compressed('glove.npz', embeddings=embeddings)
else:
    semanticspace = SemanticSpace(300, 10)
    # sparse index vectors
    # 2 output in npz form
    # Save np.array to file
#    np.savez_compressed('glove.npz', embeddings=embeddings)


# generate permutations for each pos
# permute each index vector (try both dense and sparse cases)
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






