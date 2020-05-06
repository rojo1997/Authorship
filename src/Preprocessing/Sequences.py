from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

from nltk.corpus import stopwords
from string import punctuation

import numpy as np

class Sequences(BaseEstimator, TransformerMixin):
    def __init__(self, verbose = False, num_words = 2000, maxlen = 200, 
        padding = 'post', truncating = 'post'):
        # Imprimir pasos
        self.verbose = verbose
        # Maximo de palabras
        self.num_words = num_words
        # Tokenizador
        self.tokenizer = text.Tokenizer(
            num_words = self.num_words,
            lower = True,
            split = ' '
        )
        # Maxima logitud del vector
        self.maxlen = maxlen
        # Añadir ceros
        self.padding = padding
        # Truncar frases
        self.truncating = truncating
        self.stop_words = stopwords.words("spanish") + list(punctuation)

    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        return(self)

    def transform(self, X):
        return(sequence.pad_sequences(
            sequences = self.tokenizer.texts_to_sequences(X),
            maxlen = self.maxlen,
            padding = self.padding,
            truncating = self.truncating
        ))

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
    preprocessing = Sequences(verbose = True)
    X_preprocessing = preprocessing.fit_transform(X, y)
    print(X_preprocessing)