from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation

from multiprocess import Pool
import numpy as np
import pandas as pd
import re
from googletrans import Translator

def parallel_apply(df, func, n_cores, n_jobs):
    df_split = np.array_split(df, n_jobs)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return(df)

class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'spanish'):
        self.stemmer = SnowballStemmer(language)

    def fit(self, X, y):
        return(self)

    def transform(self, X):
        if X.shape[0] < 100:
            return(X.astype(str).apply(lambda f: ' '.join([self.stemmer.stem(v) for v in f.lower().split(' ')])))
        else:
            return(parallel_apply(
                X, 
                lambda df: df.apply(lambda f: ' '.join([self.stemmer.stem(v) for v in f.lower().split(' ')])),
                n_cores = 4,
                n_jobs = 4
            ))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))


class StopWords(BaseEstimator, TransformerMixin):
    def __init__(self, language = 'spanish'):
        self.stop_words = stopwords.words(language)

    def fit(self, X, y):
        self.my_filter = re.compile('|'.join(map(lambda s: re.escape(' ' + s + ' '), self.stop_words)))
        return(self)

    def transform(self, X):
        return(X.astype(str).apply(lambda s: self.my_filter.sub(' ', s.lower())))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class Puntuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.puntuation = list(punctuation)

    def fit(self, X, y):
        self.my_filter = re.compile('|'.join(map(lambda s: re.escape(s), self.puntuation)))
        return(self)

    def transform(self, X):
        return(X.astype(str).apply(lambda s: s.replace('.','').replace(',','').replace(':','').replace(';','').replace('?','').replace('¿','')))

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

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