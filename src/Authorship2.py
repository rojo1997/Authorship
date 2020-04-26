###############################################################################


###############################################################################
import time
import numpy as np
import pandas as pd
import string
import dill as pickle

import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from NVarPrint import NVarPrint
from functions import real_xml, filter_df

class Authorship(BaseEstimator, ClassifierMixin):
    def __init__(self, le = None, random_state = 1, verbose = False):
        # Codificacion para los nombres en Y
        self.le = le
        self.epochs = 1000
        self.num_words = 5000
        self.maxlen = 200
        self.embedding_dim = 150
        self.tokenizer = Tokenizer(
            num_words = self.num_words,
            lower = True,
            split = " "
        )
        self.clf = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_length = self.maxlen, 
                input_dim = self.num_words, 
                output_dim = self.embedding_dim,
            ),
            tf.keras.layers.SpatialDropout1D(
                rate = 0.2
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.embedding_dim, 
                    dropout = 0.2, 
                    recurrent_dropout = 0.2
                )
            ),
            tf.keras.layers.Dense(
                self.embedding_dim, 
                activation = 'relu'
            ),
            tf.keras.layers.Dense(
                97, 
                activation = 'softmax'
            ),
        ])
        print(self.clf.summary())
        self.clf.compile(
            loss = 'categorical_crossentropy',
            opimizer = 'adam',
            metrics = ['accuracy', 'categorical_accuracy'],
        )
    
    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        X_dict = self.tokenizer.word_index
        print(len(X_dict))
        X_seq = self.tokenizer.texts_to_sequences(X)
        print(X_seq[:2])
        X_padded_sep = pad_sequences(
            X_seq, 
            padding = 'post', 
            maxlen = self.maxlen
        )
        print(X_padded_sep.shape)
        self.clf.fit(
            X_padded_sep, 
            y, 
            epochs = self.epochs,
            validation_split = 0.1
        )
        #self.clf.fit(X = X, y = self.le.transform(y))

    def predict(self, X):
        #return(self.le.inverse_transform(self.clf.predict(X),))
        pass

    def score(self, X, y):
        #return(self.clf.score(X,self.le.transform(y)))
        pass


def main():
    random_state = 1
    nwords = 20
    frecuency = 100

    df = real_xml('./iniciativas08/')#.sample(20000, random_state = random_state)
    print("Leido XML: ", df.shape)
    print(time.strftime("%X"))
    
    """print(time.strftime("%X"))
    df = pd.read_csv(
        "data/data.csv", 
        sep='\t', 
        encoding='utf-8',
        #nrows = 800
    )
    print(time.strftime("%X"))
    print("Leido CSV: ", df.shape)"""
    #print("Columns: ", df.columns)

    """data.to_csv("data.csv", sep='\t', encoding='utf-8', index = None)
    print("Guardado CSV")
    print(time.strftime("%X"))"""

    df = filter_df(df, nwords = nwords, frecuency = frecuency)
    print("Filtro sobre Dataset: ", df.shape)

    le = LabelEncoder()
    le.fit(df['name'])
    #print("Etiquetas codificadas: ", le.classes_)

    print('Numero de documentos: ', df.shape[0])
    print('Etiquetas: ', len(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        pd.get_dummies(df['name']).values,
        test_size = 0.2,
        random_state = random_state
    )
    print("Train: ", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)

    clf = Authorship(
        verbose = 1,
        random_state = random_state,
        le = le,
    )

    print(time.strftime("%X"))
    clf.fit(X = X_train, y = y_train)
    print(time.strftime("%X"))
    
if __name__ == "__main__":
    main()