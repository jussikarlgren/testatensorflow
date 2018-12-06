import json
from nltk import word_tokenize
import os
from logger import logger
import re

debug = True
monitor = False
error = True


def getfilelist(resourcedirectory="/home/jussi/data/recfut", pattern=re.compile(r".*008.json")):
    filenamelist = []
    for filenamecandidate in os.listdir(resourcedirectory):
        if pattern.match(filenamecandidate):
            logger(filenamecandidate, debug)
            filenamelist.append(os.path.join(resourcedirectory,filenamecandidate))
    logger(filenamelist, debug)
    return sorted(filenamelist)


def readtexts() -> list:
    filenamelist = getfilelist()
    dorawtextfiles(filenamelist, debug)


def dorawtextfiles(filenamelistunsorted, loglevel=False):
    tweetantal = 0
    sentencelist = []
    filenamelist = sorted(filenamelistunsorted)
    logger("Reading from " + str(filenamelist), loglevel)
    for filename in filenamelist:
        m = re.search(r'of_(\d+).json', filename)
        outfilebaseindex = m.group(0)
        outfiledir = '/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/'
        sl = doonetextfile(filename, outfiledir, outfilebaseindex, loglevel)
        sentencelist = sentencelist + sl
        tweetantal += len(sl)


def extractitem(tag, item):
    logger("\t{}: {}-{}\t{}\t{}".format(
        tag,
        item[tag]["textloc"]["first"],
        item[tag]["textloc"]["last"],
        item[tag]["type"],
        item[tag]["name"]), debug)
    return [item[tag]["textloc"]["first"],
        item[tag]["textloc"]["last"],
        item[tag]["type"],
        item[tag]["name"]]

def extractentity(item):
    logger("\t{}: {}-{}\t{}\t{}".format(
        "entity",
        item["textloc"]["first"],
        item["textloc"]["last"],
        item["type"],
        item["name"]), debug)
    return [item["textloc"]["first"],
        item["textloc"]["last"],
        item["type"],
        item["name"]]


def doonetextfile(filename, outfiledir, outfilebaseindex, loglevel=False):
    logger("\tReading from " + filename, loglevel)
    with open(outfiledir + "words."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "tags."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "deps."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "cats."
              + outfilebaseindex + '.txt', "w+") as outfile:
        outfile.write("")
    with open(filename, errors="replace", encoding='utf-8') as inputtextfile:
        try:
            data = json.load(inputtextfile)
            analysis = data["analysis"]
            sentences = analysis["sentences"]
            logger("Found " + str(len(sentences)), debug)
            for a in sentences:
                foundstuff = []
                if "references" in a:
                    for e in a["references"]:
                        if "event" in e:
                            if "attributes" in e["event"]:
                                if "target" in e["event"]["attributes"]:
                                    try:
                                        if type(e["event"]["attributes"]["target"]) == list:
                                            for vv in e["event"]["attributes"]:
                                                foundstuff.append(extractitem("target", vv))
                                        else:
                                            foundstuff.append(extractitem("target", e["event"]["attributes"]))
                                    except:
                                        logger("\t****" + str(e["event"]["attributes"]["target"]), error)
                                if "attacker" in e["event"]["attributes"]:
                                    try:
                                        if type(e["event"]["attributes"]["attacker"]) == list:
                                            for vv in e["event"]["attributes"]:
                                                foundstuff.append(extractitem("attacker", vv))
                                        else:
                                            foundstuff(extractitem("attacker", e["event"]["attributes"]))
                                    except:
                                        logger("\t****" + str(e["event"]["attributes"]["attacker"]), error)
                                if "entities" in e["event"]["attributes"]:
                                    if type(e["event"]["attributes"]["entities"]) == list:
                                        for eee in e["event"]["attributes"]["entities"]:
                                                foundstuff.append(extractentity(eee))
                                    else:
                                        foundstuff.append(extractentity(e["event"]["attributes"]["entities"]))
                z = {}
                for fs in foundstuff:
                    logger(str(fs), debug)
                    bio = "B"
                    for zz in range(fs[0], fs[1] + 1):
                        z[zz] = fs.copy()
                        z[zz][2] = bio + "-" + fs[2]
                        bio = "I"
#                for t in a["depgraph"]:
#                    if " " in t["node_labels"]["FORM"]:
#                        form = t["node_labels"]["FORM"].split(" ")
#                    elif "/" in t["node_labels"]["FORM"]:
#                        form = t["node_labels"]["FORM"].split("/")
#                        form = [elem for x in form for elem in (x, "/")][:-1]
#                    elif t["node_labels"]["FORM"] == "--":
#                        form = ["-","-"]
#                    elif "-" in t["node_labels"]["FORM"] and len(t["node_labels"]["FORM"]) > 1:
#                        form = t["node_labels"]["FORM"].split("-")
#                        form = [elem for x in form for elem in (x, "-")][:-1]
#                    else:
#                        form = [t["node_labels"]["FORM"]]
#                    x[int(t["node_labels"]["ID"])] = []
#                    y[int(t["node_labels"]["ID"])] = []
#                    for fs in form:
#                        x[int(t["node_labels"]["ID"])].append(t)
#                        y[int(t["node_labels"]["ID"])].append(fs)
                tokenout = ""
                facitout = ""
                depout = ""
                tagout = ""
                ix = 0
                tokens = a["tokens"]
                for ttt in tokens:
                    bio = "OUT"
                    w = ttt["w"]
                    n = ttt["w"]
                    p = ttt["pos"]
                    if ix in z:
                        bio = z[ix][2]
                        n = z[ix][3]
                    logger("e:\t{}\t{}\t{}\t{}\t{}".format(ix, w, n, p, bio), debug)
                    tokenout += " " + ttt["w"]
                    facitout += " " + bio
                    tagout += " " + p
                    ix += 1
                with open('/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/words.'
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(tokenout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/tags."
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(tagout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/deps."
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(depout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/cats."
                          + outfilebaseindex + '.txt', "a+") as outfile:
                    outfile.write(facitout + "\n")
        except json.decoder.JSONDecodeError as e:
            logger("***\t" + filename + "\t" + str(e.msg), error)
            print(e)
            data = []
    return sentences


readtexts()

# fixa facit
# create 300-dim ri vectors, multiply with dense pos vectors