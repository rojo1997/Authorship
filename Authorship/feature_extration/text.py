from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

class Sequences(BaseEstimator, TransformerMixin):
    def __init__(self, num_words = 2000, maxlen = 200, padding = 'post', 
        truncating = 'post'):
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