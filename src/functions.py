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
import xml.etree.ElementTree as ET

stemmer = SnowballStemmer('spanish')

def clean(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return (re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text))

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

def filter_df(df, nwords = 3, frecuency = 10):
    df['nwords'] = df['text'].apply(lambda s: len(s.split(' ')))
    print("antes: ", df.shape)
    df.drop(df[df['nwords'] <= nwords].index, inplace = True)
    print("despues: ", df.shape)
    groups = df['name'].value_counts()
    df["gr"] = df.name.map(lambda n: float(groups.get(key = n)))
    df.drop(df[df.gr <= frecuency].index, inplace = True)
    print('despues: ', df.shape)
    return(df)

def real_xml(path = 'iniciativas08/'):
    files = []
    df = []
    for _,_,f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(path + file)
    files = pd.DataFrame(files, columns = ['file'])#.head(5)
    #print(files)

    files['intervenciones'] = files['file'].apply(
        lambda f: ET.parse(f).findall('iniciativa/intervencion')
    )
    
    for _,intervenciones in files['intervenciones'].iteritems():
        for intervencion in intervenciones:
            df.append({
                'name': intervencion.findtext('interviniente'),
                'text': '. '.join([str(parrafo.text) for parrafo in intervencion.findall('discurso/parrafo')])
            })
    df = pd.DataFrame(df, columns = ['name','text'])

    filters = [line.replace('\n', '').strip() for line in open('replace.txt', 'r', encoding = 'utf8').readlines()]

    my_filter = re.compile('|'.join(map(re.escape, filters)))

    df['name'] = df['name'].apply(lambda name: 
        name.replace(
            "PÉREZ GARCÍA DE PRADO", "PÉREZ GARCÍA DEL PRADO"
        ).replace(
            ',', ''
        ).replace(
            '.', ''
        ).replace(
            '/', ''
        ).strip()
    )

    

    df['name'] = df['name'].apply(lambda name:
        my_filter.sub("", name).strip()
    )

    df.drop(df[df['name'] == ''].index,inplace = True)

    df['text'] = df['text'].apply(lambda text: clean(text.replace('-','')))

    print(df['name'].unique())
    df['name'] = df['name'].apply(lambda name: name.upper())
    df['name'] = df['name'].apply(lambda name: ' '.join(name.split(' ')[:4]))

    print(df['name'].unique().shape)
    print(df['name'].value_counts())
    print(df.shape)

    return(df)