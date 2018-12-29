"""Build an np.array for a text file with an accompanying tag file using random indexing and overlay vector addition.
   Intended to be used with Guillaume's tensorflow sequence predictor.

"""

__author__ = "Jussi Karlgren"

import sparsevectors
import numpy as np
import os
from logger import logger
from collections import Counter

error = True
monitor = True
debug = True

dimensionality = 300
densedensity = 150
sparsedensity = 10

density = densedensity
labeldensity = sparsedensity

path = "/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/"
outpath = "/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/"


def mergefiles(wordfile, posfile, depfile, nerfile):
    struct = []
    with open(wordfile,'r+') as f_words, open(posfile,'r+') as f_pos, open(depfile,"r+") as f_deps, open(nerfile,"r+") as f_ners:
        for line_words, line_pos, line_deps, line_ners in zip(f_words, f_pos, f_deps, f_ners):
            w1 = line_words.split()
            p1 = line_pos.split()
            d1 = line_deps.split()
            n1 = line_ners.split()
            struct.append(zip(w1, p1, d1, n1))
    return struct

categories = ["wordtokens", "postags", "dependencies", "ner"]
MINCOUNT = 1
joinstring = "#%#"

def doallthefiles(rangelimit=4000):
    filelist = {}
    seenfile = {}
    antal_frag = 0
    for ix in range(rangelimit):
        filelist[ix] = {}
        seenfile[ix] = True
        for cat in categories:
            fn = "{}{}.of_{:0>4d}.json.txt".format(path, cat, ix)
            try:
                os.stat(fn)
                filelist[ix][cat] = fn
            except:
                seenfile[ix] = None
                filelist[ix][cat] = None
                del filelist[ix]
                logger("index {} did not match up {} file: {}".format(ix, cat, fn), error)
    logger("antal filer: {}".format(len(filelist)), monitor)
    conditions = ["wp", "wd", "wn", "wdp", "wnp", "wnd", "wndp"]
    vocabulary = {}
    vocabulary_words = Counter()
    vocabulary_labels = Counter()
    vocabulary["wp"] = Counter()
    vocabulary["wd"] = Counter()
    vocabulary["wn"] = Counter()
    vocabulary["wnp"] = Counter()
    vocabulary["wnd"] = Counter()
    vocabulary["wdp"] = Counter()
    vocabulary["wndp"] = Counter()
    outfrag = {}
    for fileindex in filelist:
        if seenfile[fileindex]:
            zippy = mergefiles(filelist[fileindex][categories[0]], filelist[fileindex][categories[1]],
                               filelist[fileindex][categories[2]], filelist[fileindex][categories[3]])
            wp_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wp", fileindex), "w+")
            wd_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wd", fileindex), "w+")
            wn_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wn", fileindex), "w+")
            wnp_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wnp", fileindex), "w+")
            wnd_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wnd", fileindex), "w+")
            wdp_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wdp", fileindex), "w+")
            wndp_f = open('{}{}/new_{:0>4d}.txt'.format(outpath, "wndp", fileindex), "w+")
            for fragment in zippy:
                antal_frag += 1
                for cc in conditions:
                    outfrag[cc] = []
                for oneitem in fragment:
                    vocabulary_words.update([oneitem[0]])
                    vocabulary_labels.update([oneitem[1]])
                    vocabulary_labels.update([oneitem[2]])
                    vocabulary_labels.update([oneitem[3]])
                    vocabulary["wp"].update([joinstring.join([oneitem[0], oneitem[1]])])
                    outfrag["wp"].append("".join([oneitem[0], oneitem[1]]))
                    vocabulary["wd"].update([joinstring.join([oneitem[0], oneitem[2]])])
                    outfrag["wd"].append("".join([oneitem[0], oneitem[2]]))
                    vocabulary["wn"].update([joinstring.join([oneitem[0], oneitem[3]])])
                    outfrag["wn"].append("".join([oneitem[0], oneitem[3]]))
                    vocabulary["wnp"].update([joinstring.join([oneitem[0], oneitem[1], oneitem[2]])])
                    outfrag["wnp"].append("".join([oneitem[0], oneitem[1], oneitem[2]]))
                    vocabulary["wnd"].update([joinstring.join([oneitem[0], oneitem[1], oneitem[3]])])
                    outfrag["wnd"].append("".join([oneitem[0], oneitem[1], oneitem[3]]))
                    vocabulary["wdp"].update([joinstring.join([oneitem[0], oneitem[2], oneitem[3]])])
                    outfrag["wdp"].append("".join([oneitem[0], oneitem[2], oneitem[3]]))
                    vocabulary["wndp"].update([joinstring.join([oneitem[0], oneitem[1], oneitem[2], oneitem[3]])])
                    outfrag["wndp"].append("".join([oneitem[0], oneitem[1], oneitem[2], oneitem[3]]))
                wp_f.write(" ".join(outfrag["wp"]) + "\n")
                wd_f.write(" ".join(outfrag["wd"]) + "\n")
                wn_f.write(" ".join(outfrag["wn"]) + "\n")
                wnp_f.write(" ".join(outfrag["wnp"]) + "\n")
                wnd_f.write(" ".join(outfrag["wnd"]) + "\n")
                wdp_f.write(" ".join(outfrag["wdp"]) + "\n")
                wndp_f.write(" ".join(outfrag["wndp"]) + "\n")
            wn_f.close()
            wd_f.close()
            wp_f.close()
            wnd_f.close()
            wnp_f.close()
            wdp_f.close()
            wndp_f.close()

    logger("antal fragment: {}".format(antal_frag), monitor)
    vocab_words = {w for w, c in vocabulary_words.items() if c >= MINCOUNT}
    size_vocab = len(vocab_words)
    logger("antal ord std: {}".format(size_vocab), monitor)
    embeddings = {}
    for w in vocab_words:
        embeddings[w] = sparsevectors.newrandomvector(dimensionality, density)

    vocab_labels = {w for w, c in vocabulary_labels.items() if c >= MINCOUNT}
    size_vocab = len(vocab_labels)
    logger("antal tag tot: {}".format(size_vocab), monitor)
    labelembeddings = {}
    for w in vocab_labels:
        try:
            labelembeddings[w] = sparsevectors.newrandomvector(dimensionality, labeldensity)
        except IndexError:
            logger("Indexerror: {}".format(w), error)
    for cc in conditions:
        vocab_words = {w for w, c in vocabulary[cc].items() if c >= MINCOUNT}
        size_vocab = len(vocab_words)
        compositeembeddings = {}
        logger("antal ord i {}: {}".format(cc, size_vocab), monitor)
        with open('{}{}/vocab.words.txt'.format(outpath, cc),"w+") as f:
            for wdl in sorted(list(vocab_words)):
                wd = "".join(wdl.split(joinstring))
                f.write('{}\n'.format(wd))
                vv = embeddings[wdl.split(joinstring)[0]]
                for ll in wdl.split(joinstring)[1:]:
                    vv = sparsevectors.sparseadd(vv, labelembeddings[ll])
                compositeembeddings[wd] = sparsevectors.listify(sparsevectors.normalise(vv), dimensionality)
        with open('{}{}/compositevectors.txt'.format(outpath, cc), "w+") as f:
            for www in compositeembeddings:
                f.write("{} {}\n".format(www, " ".join(map(str, compositeembeddings[www]))))


howmany = 2937
# read words file
if __name__ == '__main__':


    doallthefiles(howmany)

