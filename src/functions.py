from xml.dom import minidom

import os
import time
import re
import platform
from tqdm import tqdm 

from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')

def readDocument(docname):
    # We open de document
    mydoc = minidom.parse(docname)
    # We request the list of interventions
    intervencion = mydoc.getElementsByTagName('intervencion')
    # Data structure
    data = pd.DataFrame({'name':[],'text':[]})
    # Detect lowercase
    lower = re.compile(r'.*[a-z]+')

    for elem in intervencion:
        nodes = elem.childNodes
        name = (nodes[1].childNodes[0].nodeValue).replace("El señor ","")
        name = name.replace("La señora ","")
        name = name.replace("El señora ","")
        name = name.replace("-","")
        name = name.replace("_","")
        pos = name.find(',')
        if pos != -1:
            name = name[0:(pos)]

        if "la palabra" in name: continue
        if '.' in name: continue
        if "VICEPRESIDENT" in name: continue
        if "PRESIDENT" in name: continue
        if "REPRESENTANTE" in name: continue
        if "LETRADO" in name: continue
        if "CONSEJER" in name: continue
        if lower.match(name): continue

        parrafos = nodes[3].childNodes

        text = ""
        for parrafo in parrafos:
            if parrafo.nodeType == 1:
                if len(parrafo.childNodes):
                    text = text + " " + parrafo.childNodes[0].nodeValue
        text = text.replace("-","")
        text = text.replace("_","")
        text = text.replace("[...]","")
        if text: data = data.append({'name':name,'text':text},ignore_index=True)
    return (data)

def readData(path):
    files = []
    data = pd.DataFrame({'name':[],'text':[]})
    for _,_,f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(path + file)
    files.sort()
    data = pd.concat([readDocument(docname) for docname in files])
    return(data)

def nphrases(str):
    p = str.count('.')
    if p == 0: return (1)
    else: return (p)

def tag(words):
    pairs = pos_tag(words)
    sett = []
    for _,j in pairs:
        sett.append(j)
    return(' '.join(sett))

def tupleN(words, n):
    string = ""
    for i in range(0,len(words) - n,n):
        string += ('_'.join(words[i:(i+n)]) + ' ')
    return (string)

def stem(w):
    return([stemmer.stem(word.lower()) for word in w])

def feature_extraction (data):
    upper = re.compile(r'.*[A-Z]+')
    ss = SnowballStemmer('spanish')

    tqdm.pandas(desc="Tokens Bar")
    data['token'] = data['text'].progress_map(word_tokenize)
    tqdm.pandas(desc="Roots Bar")
    data['root'] = data['token'].progress_map(stem)
    tqdm.pandas(desc="Analysisw Bar")
    data['analysisw'] = data['token'].progress_map(tag)
    data['nchars'] = data['text'].map(len)
    print("nchars")
    data['nupper'] = data['text'].map(lambda w: np.sum(map(upper.match,w)))
    print("nupper")
    data['nphrases'] = data['text'].map(nphrases)
    print("nphrases")
    data['nwords'] = data['token'].map(len)
    print("nwords")
    data['nwperp'] = data['nwords'] / data['nphrases']
    print("nwperp")
    

    levels = 3
    data['levels_root'] = ""
    for i in range(2, levels + 1):
        data['levels_root'] += data['root'].map(lambda ws: tupleN(ws,i)) + ' '
        print(i)
    data['levels_analy'] = ""
    for i in range(2, levels + 1):
        data['levels_analy'] += data['analysisw'].map(lambda ws: tupleN(ws,i)) + ' '
        print(i)
    data['levels_token'] = ""
    for i in range(2, levels + 1):
        data['levels_token'] += data['token'].map(lambda ws: tupleN(ws,i)) + ' '
        print(i)