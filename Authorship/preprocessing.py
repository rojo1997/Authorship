from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, pos_tag_sents, word_tokenize
from string import punctuation

from multiprocess import Pool
import numpy as np
import pandas as pd
import re
from googletrans import Translator

min_parallel = 200

def parallel_apply(df, func, n_cores, n_jobs):
    df_split = np.array_split(df, n_jobs)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return(df)

class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'spanish', model = 'snowball'):
        if model.lower() == 'porter': 
            self.stemmer = PorterStemmer()
        if model.lower() == 'snowball':
            self.stemmer = SnowballStemmer(language)

    def fit(self, X, y):
        return(self)

    def transform(self, X):
        if X.shape[0] < min_parallel:
            return(X.astype(str).apply(lambda f: ' '.join([self.stemmer.stem(v) for v in word_tokenize(f.lower())])))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.apply(lambda f: ' '.join([self.stemmer.stem(v) for v in word_tokenize(f.lower())])),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, pos = 'a'):
        self.lemmatizer = WordNetLemmatizer()
        self.pos = pos

    def fit(self, X, y):
        return(self)

    def transform(self, X):
        if X.shape[0] < min_parallel:
            return(X.astype(str).apply(lambda f: ' '.join([self.lemmatizer.lemmatize(v, self.pos) for v in f.lower().split(' ')])))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.apply(lambda f: ' '.join([self.lemmatizer.lemmatize(v, self.pos) for v in f.lower().split(' ')])),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class StopWords(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'spanish'):
        self.stop_words = set(stopwords.words(language))

    def fit(self, X, y):
        return(self)

    def transform(self, X):
        if X.shape[0] < min_parallel:
            return(X.astype(str).apply(lambda s: ' '.join([s for s in s.split(' ') if s.lower() not in self.stop_words])))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.astype(str).apply(lambda s: ' '.join([s for s in s.split(' ') if s.lower() not in self.stop_words])),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class Puntuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.puntuation = list(punctuation) + ['¿','¡']

    def fit(self, X, y):
        self.my_filter = re.compile('|'.join(map(lambda s: re.escape(s), self.puntuation)))
        return(self)

    def transform(self, X):
        if X.shape[0] < min_parallel:
            return(X.astype(str).apply(self.replace))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.astype(str).apply(self.replace),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

    def replace(self, text):
        for p in self.puntuation:
            text = text.replace(p, '')
        return(text)

class Translate(BaseEstimator, TransformerMixin):
    def __init__(self, src = 'es', dest = 'en'):
        self.src = 'es'
        self.dest = 'en'

    def fit(self, X, y):
        pass

    def transform(self, X):
        translator = Translator()
        return(
            X.astype(str).apply(
                lambda s: '.'.join(
                    [translator.translate(f, src = self.src, dest = self.dest).text for f in s.split('.') if f.strip() != '']
                )
            )
        )

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class PosTag(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'english', lang = 'eng'):
        self.language = language
        self.lang = lang

    def fit(self, X, y):
        pass

    def transform(self, X):
        if X.shape[0] < min_parallel:
            return(X.astype(str).apply(lambda s: ' '.join([tag for v,tag in pos_tag(word_tokenize(s,language = self.language), lang = self.lang)])))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.astype(str).apply(lambda s: ' '.join([tag for v,tag in pos_tag(word_tokenize(s,language = self.language), lang = self.lang)])),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))