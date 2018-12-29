import json
from nltk import word_tokenize
import os
from logger import logger
import re

debug = False
monitor = True
error = True


def getfilelist(resourcedirectory="/home/jussi/data/recfut/merdata/cyber", pattern=re.compile(r".*.json")):
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
        try:
            sl = doonetextfile(filename, outfiledir, outfilebaseindex)
            sentencelist = sentencelist + sl
            tweetantal += len(sl)
        except KeyError as ee:
            logger("Error in {}".format(filename), error)
            logger(str(ee), error)


def extractentity(tag, item, loglevel=False):
    flip = generalise(item["type"])
    logger("\t{}: {}-{}\t{}->{}\t{}".format(
        tag,
        item["textloc"]["first"],
        item["textloc"]["last"],
        item["type"],
        flip,
        item["name"]), loglevel)
    return [item["textloc"]["first"],
        item["textloc"]["last"],
        flip,
        item["name"],
        tag]

def generalise(tag):
    if tag == "City":
        return "GeoEntity"
    elif tag == "Region":
        return "GeoEntity"
    elif tag == "Country":
        return "GeoEntity"
    elif tag == "ProvinceOrState":
        return "GeoEntity"

    elif tag == "Airport":
        return "Industry"
    elif tag == "TVStation":
        return "Industry"
    elif tag == "Facility":
        return "Industry"
    elif tag == "Airport":
        return "Industry"

    elif tag == "Company":
        return "Organization"
    elif tag == "OrgEntity":
        return "Organization"
    elif tag == "Religion":
        return "Organization"
    elif tag == "PublishedMedium":
        return "Organization"


    elif tag == "InternetDomainName":
        return "URL"
    elif tag == "IpAddress":
        return "URL"

    elif tag == "Malware":
        return "AttackVector"
    elif tag == "MalwareCategory":
        return "AttackVector"

    elif tag == "Product":
        return "IndustryTerm"
    elif tag == "Technology":
        return "IndustryTerm"
    elif tag == "TechnologyArea":
        return "IndustryTerm"
    return tag


def doonetextfile(filename, outfiledir, outfilebaseindex):
    logger("\tReading from " + filename, monitor)
    with open(outfiledir + "wordtokens."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "postags."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "dependencies."
              + outfilebaseindex + '.txt', 'w+') as outfile:
        outfile.write("")
    with open(outfiledir + "ner."
              + outfilebaseindex + '.txt', "w+") as outfile:
        outfile.write("")
    with open(outfiledir + "entities."
              + outfilebaseindex + '.txt', "w+") as outfile:
        outfile.write("")
    with open(filename, errors="replace", encoding='utf-8') as inputtextfile:
        try:
            data = json.load(inputtextfile)
            analysis = data["analysis"]
            sentences = analysis["sentences"]
            logger("Found {} sentences".format(len(sentences)), monitor)
            for a in sentences:
                foundstuff = []
                if "references" in a:
                    for e in a["references"]:
                        if "event" in e:
                            if "attributes" in e["event"]:
                                if "target" in e["event"]["attributes"]:
                                    #try:
                                    if type(e["event"]["attributes"]["target"]) == list:
                                        for vv in e["event"]["attributes"]["target"]:
                                            if type(vv) != str:
                                                foundstuff.append(extractentity("target", vv))
                                            else:
                                                logger("Strange attribute {}".format(vv), debug)
                                                if vv == "target":
                                                    logger(str(e["event"]["attributes"]), debug)
                                    else:
                                        foundstuff.append(extractentity("target", e["event"]["attributes"]["target"]))
                                    #except:
                                    #    logger("\t**!**" + str(e["event"]["attributes"]["target"]), error)
                                    #    logger("\t**>**" + str(e["event"]["attributes"]), error)
                                if "attacker" in e["event"]["attributes"]:
                                    try:
                                        if type(e["event"]["attributes"]["attacker"]) == list:
                                            for vv in e["event"]["attributes"]["attacker"]:
                                                if type(vv) != str:
                                                    foundstuff.append(extractentity("attacker", vv, debug))
                                                else:
                                                    logger("Strange attribute {}".format(vv), True)
                                        else:
                                            foundstuff.append(extractentity("attacker", e["event"]["attributes"]["attacker"]))
                                    except:
                                        logger("\t**x**" + str(e["event"]["attributes"]["attacker"]), error)
                                if "entities" in e["event"]["attributes"]:
                                    if type(e["event"]["attributes"]["entities"]) == list:
                                        for eee in e["event"]["attributes"]["entities"]:
                                                foundstuff.append(extractentity("entity", eee))
                                    else:
                                        foundstuff.append(extractentity("entity", e["event"]["attributes"]["entities"]))
                nersamling = {}
                recfutnamesamling = {}
                recfuttagsamling = {}
                for fs in foundstuff:
                    bio = "B"
                    for zz in range(fs[0], fs[1] + 1):
                        if zz in recfuttagsamling:
                            if (fs[4] == "target" or fs[4] == "attacker") and recfuttagsamling[zz].endswith("entity"):
                                logger("{} -> {}".format(recfuttagsamling[zz], fs), debug)
                                recfuttagsamling[zz] = bio + "-" + fs[4]
                        else:
                            recfuttagsamling[zz] = bio + "-" + fs[4]
                        nersamling[zz] = bio + "-" + fs[2]
                        recfutnamesamling[zz] = fs[3]
                        bio = "I"
                d = {}
                df = {}
                dh = {}
                for t in a["depgraph"]:
                    head = t["head"]
                    dep = t["edge_labels"]["DEPREL"]
                    form = t["node_labels"]["FORM"]
                    id = int(t["node_labels"]["ID"]) - 1
                    logger("d:\t{}-{}->\t{}\t{}".format(id, dep, head, form), debug)
                    df[id] = form
                    d[id] = dep
                    dh[id] = head
                    # #                    if " " in t["node_labels"]["FORM"]:
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
#                    x[int(t["node_labels"]["ID"])] = []
#                    y[int(t["node_labels"]["ID"])] = []
#                    for fs in form:
#                        x[int(t["node_labels"]["ID"])].append(t)
#                        y[int(t["node_labels"]["ID"])].append(fs)
                tokenout = ""
                recfutentitiesout = ""
                dependenciesout = ""
                namedentities = ""
                posout = ""
                ix = 0
                ixd = 0
                dependencypanic = False
                tokens = a["tokens"]
                for ttt in tokens:
                    recfuttag = "O"
                    wordtoken = ttt["w"]
                    name = ttt["w"]
                    pos = ttt["pos"]
                    if ix in recfuttagsamling:
                        recfuttag = recfuttagsamling[ix]
                        name = recfutnamesamling[ix]
                    if ix in nersamling:
                        nertag = nersamling[ix]
                    else:
                        nertag = "0"
                    dwd = "."
                    dep = "NIL"
                    head = "."
                    if ixd in d:
                        dep = d[ixd]
                        dwd = df[ixd]
                        head = dh[ixd]
                    try:
                        if dwd != wordtoken and len(tokens) > ixd and tokens[ixd + 1]["w"] == dwd:
                            ixd += 1
                            dep = d[ixd]
                            dwd = df[ixd]
                            head = dh[ixd]
                    except:
                        logger("\t\t{} {} {} {}".format(dwd,ix,ixd,len(tokens)))
                    if dwd != wordtoken:
                        logger("\t****panic: mismatch:", debug)
                        dependencypanic = True
                    tokenout += " " + ttt["w"]
                    recfutentitiesout += " " + recfuttag
                    namedentities += " " + nertag
                    posout += " " + pos
                    if dependencypanic:
                        if dwd == wordtoken:
                            dependencypanic = False
                        else:
                            dep = "NIL"
                    dependenciesout += " " + dep
                    logger("e:\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}->{}".format(ix, wordtoken, name, pos, recfuttag, nertag, ixd, dwd, dep, head), debug)
                    ix += 1
                    ixd += 1
                with open('/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/wordtokens.'
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(tokenout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/postags."
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(posout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/dependencies."
                          + outfilebaseindex + '.txt', 'a+') as outfile:
                    outfile.write(dependenciesout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/entities."
                          + outfilebaseindex + '.txt', "a+") as outfile:
                    outfile.write(recfutentitiesout + "\n")
                with open("/home/jussi/aktuellt/2018.recfut/tf_ner/data/recfut/new/ner."
                          + outfilebaseindex + '.txt', "a+") as outfile:
                    outfile.write(namedentities + "\n")
        except json.decoder.JSONDecodeError as e:
            logger("***\t" + filename + "\t" + str(e.msg), error)
            print(e)
            data = []
    return sentences


readtexts()

# fixa facit
# create 300-dim ri vectors, multiply with dense pos vectors