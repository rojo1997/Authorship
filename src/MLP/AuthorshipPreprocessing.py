from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

from nltk.corpus import stopwords
from string import punctuation

import numpy as np

class AuthorshipPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, verbose = False, k = 10, ngram_range = (1,3), 
        max_features = 50):
        # Imprimir pasos
        self.verbose = verbose
        # Maximo de caracteristicas
        self.k = k
        # Combinatoria de palabras
        self.ngram_range = ngram_range
        # Maximo de caracteristicas sobre texto
        self.max_features = max_features
        self.stop_words = stopwords.words("spanish") + list(punctuation)

        self.preprocessing = Pipeline([
            ('TfidfVectorizer', TfidfVectorizer(
                stop_words = self.stop_words,
                analyzer = 'word',
                encoding = 'utf8',
                dtype = np.float32,
                ngram_range = self.ngram_range, # (1,3): 0.792
                max_features = self.max_features,
                min_df = 1.0 / 1000.0,
                max_df = 999.0 / 1000.0,
                strip_accents = 'unicode',
                decode_error = 'replace',
                lowercase = True
            )),
            ('SelectKBest', SelectKBest(
                score_func = f_classif, 
                k = self.k
            ))
        ], verbose = self.verbose)

    def fit(self, X, y):
        self.preprocessing.fit(X,y)
        return(self)

    def transform(self, X):
        X = self.preprocessing.transform(X)
        X.sort_indices()
        return(X)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

if __name__ == "__main__":
    X = [
        'hola que tal',
        'buenos dias',
        'yo estoy bien',
        'me gustaría conocerte',
        'podemos quedar un dia',
        'te viene bien el jueves de la semana que viene',
        'vale perfecto',
        'ha sido un placer'
    ]
    y = [
        'Alberto',
        'Pablo',
        'Alberto',
        'Pablo',
        'Alberto',
        'Pablo',
        'Alberto',
        'Pablo',
    ]
    preprocessing = AuthorshipPreprocessing(verbose = True)
    X_preprocessing = preprocessing.fit_transform(X, y)
    print(X_preprocessing)