from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation

from multiprocessing import Pool
import numpy as np
import pandas as pd
import re

def parallel_apply(df, func, n_cores, n_jobs):
    df_split = np.array_split(df, n_jobs)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return(df)

def replaces(cad):
            for sw in self.stop_words:
                cad = cad.replace(' ' + sw + ' ', ' ')
            return(cad)

class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'spanish'):
        self.stemmer = SnowballStemmer(language)

    def fit(self, X, y):
        return(self)

    def transform(self, X):
        return(X.astype(str).apply(lambda f: ' '.join([self.stemmer.stem(v) for v in f.split(' ')])))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))


class StopWords(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words = None):
        self.stop_words = stopwords.words("spanish") + list(punctuation) if stop_words == None else stop_words

    def fit(self, X, y):
        self.my_filter = re.compile('|'.join(map(lambda s: ' ' + re.escape(s) + ' ', self.stop_words)))
        return(self)

    def transform(self, X):
        return(X.astype(str).apply(lambda s: self.my_filter.sub(' ', s)))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))