import json
from nltk import word_tokenize
import os
from logger import logger
import re

debug = True
monitor = False
error = True


def getfilelist(resourcedirectory="/home/jussi/data/recfut", pattern=re.compile(r".*json")):
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
        sl = doonetextfile(filename, loglevel)
        sentencelist = sentencelist + sl
        tweetantal += len(sl)


def doonetextfile(filename, loglevel=False):
    logger("\tReading from " + filename, loglevel)
    sentencelist = []
    with open(filename, errors="replace", encoding='utf-8') as inputtextfile:
        try:
            data = json.load(inputtextfile)
            analysis = data["analysis"]
            sentences = analysis["sentences"]
            logger("Found " + str(len(sentences)), debug)
            for a in sentences:
                x = {}
                y = {}
                z = {}
                zp = {}
                try:
                    for e in a["entities"]:
                        logger("\t{}\t{}\t{}".format(e["type"], e["name"], e["textloc"]["first"], e["textloc"]["last"]), debug)
                        logger("\t\t{}\t{}".format(str(a["tokens"][e["textloc"]["first"]]["w"]),str(a["tokens"][e["textloc"]["first"]]["pos"])), debug)
                        for zz in range(e["textloc"]["first"], e["textloc"]["last"]):
                            z[zz] = a["tokens"][str(zz)]["w"]
                            zp[zz] = a["tokens"][str(zz)]["pos"]
                except:
                    logger("No entities found.", debug)
                for t in a["depgraph"]:
                    if " " in t["node_labels"]["FORM"]:
                        form = t["node_labels"]["FORM"].split(" ")
                    elif "/" in t["node_labels"]["FORM"]:
                        form = t["node_labels"]["FORM"].split("/")
                        form = [elem for x in form for elem in (x, "/")][:-1]
                    elif t["node_labels"]["FORM"] == "--":
                        form = ["-","-"]
                    elif "-" in t["node_labels"]["FORM"] and len(t["node_labels"]["FORM"]) > 1:
                        form = t["node_labels"]["FORM"].split("-")
                        form = [elem for x in form for elem in (x, "-")][:-1]
                    else:
                        form = [t["node_labels"]["FORM"]]
                    x[int(t["node_labels"]["ID"])] = []
                    y[int(t["node_labels"]["ID"])] = []
                    for fs in form:
                        x[int(t["node_labels"]["ID"])].append(t)
                        y[int(t["node_labels"]["ID"])].append(fs)
                tokens = a["tokens"]
                tokenlist = []
                tokenout = ""
                tagout = ""
                facitout = ""
                for ttt in tokens:
#                    if ttt["w"] == "-":
#                        continue
                    tokenlist.append([ttt["w"],ttt["pos"]])
                ix = 0
                inside = False
                for i in sorted(y.keys()):
                    for w in y[i]:
                        try:
                            if (tokenlist[ix][0] == w):
                                if inside and str(x[i][0]["node_labels"]["NAMEPOS"]).startswith("B-"):
                                    bio = str(x[i][0]["node_labels"]["NAMEPOS"]).replace("B-","I-")
                                else:
                                    bio = x[i][0]["node_labels"]["NAMEPOS"]
                                    inside = False
                                if str(x[i][0]["node_labels"]["NAMEPOS"]).startswith("B-"):
                                    inside = True
                                logger("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                                    i,
                                    w,
                                    x[i][0]["node_labels"]["FORM"],
                                    tokenlist[ix][0],
                                    tokenlist[ix][1],
                                    x[i][0]["edge_labels"]["DEPREL"],
                                    x[i][0]["head"],
                                    x[i][0]["node_labels"]["NAMECAT"],
                                    x[i][0]["node_labels"]["NAMEIND"],
                                    bio), debug)
                                tokenout += " " + w
                                tagout += " " + tokenlist[ix][1]
                                facitout += " " + bio
#                                print(x[i])
                       # for ww in w:
                       #     print("\t{}".format(ww))
                       # print("\n")
                       # print("\t{}".format(x[i]))
#                        else:
#                            print("\t\t**********{}".format(tokenlist[ix][0]))
                            ix += 1
                        except:
                            ix = 0
                with open('/home/jussi/aktuellt/2018.recfut/words.txt', 'a+') as outfile:
                    outfile.write(tokenout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tags.txt", "a+") as outfile:
                    outfile.write(tagout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/cats.txt", "a+") as outfile:
                    outfile.write(facitout + "\n")
        except json.decoder.JSONDecodeError as e:
            logger("***\t" + filename + "\t" + str(e.msg), error)
            print(e)
            data = []
    return sentences


readtexts()

# fixa facit
# create 300-dim ri vectors, multiply with dense pos vectors