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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from nltk.stem.snowball import SpanishStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from NVarPrint import NVarPrint
from functions import real_xml, filter_df

class Authorship(BaseEstimator, ClassifierMixin):
    def __init__(self, le = None, random_state = 1, verbose = False, ntarget = 0):
        # Extrator de raices en castellano
        self.stemmer = SpanishStemmer()
        # Analizador externo
        self.analyzer = CountVectorizer().build_analyzer()
        # Codificacion para los nombres en Y
        self.le = le
        self.random_state = random_state
        self.epochs = 150
        self.embedding_dim = 400
        self.max_features = 5000
        self.n_components = 500
        self.preprocessing = Pipeline([
            ('CountVectorizer', CountVectorizer(
                # Pasar minuculas
                lowercase = True,
                # Palabras que no se valoran para el modelos
                stop_words = stopwords.words("spanish") + list(punctuation),
                # Extraer la raiz de cada palabra
                analyzer = lambda doc: map(self.stemmer.stem,self.analyzer(doc)),
                # Dividir un parrafo en palabras
                tokenizer = word_tokenize,
                # Marca a 1 todos los 0
                binary = False,
                # Typo de dato para las columnas
                dtype = np.int64,
                max_features = self.max_features,
                ngram_range = (1,3)
            )),
            ('TfidfTransformer', TfidfTransformer(
                # Normalizacion l2 suma de los cuadrados del vector
                norm = "l2",
                # Frecuencia de documentos inversa
                use_idf = True,
                # Alisado de documentos
                smooth_idf = True,
                # Escala sublinear no usada
                sublinear_tf = False
            )),
            ('TruncatedSVD', TruncatedSVD(
                n_components = self.n_components,
                n_iter = 20,
                random_state = self.random_state
            ))
        ], verbose = True)

        self.clf = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.embedding_dim,
                input_shape = (self.n_components,),
                activation = 'relu'
            ),
            tf.keras.layers.Dropout(
                rate = 0.4
            ),
            tf.keras.layers.Dense(
                self.embedding_dim, 
                activation = 'relu'
            ),
            tf.keras.layers.Dropout(
                rate = 0.4
            ),
            tf.keras.layers.Dense(
                self.embedding_dim, 
                activation = 'relu'
            ),
            tf.keras.layers.Dropout(
                rate = 0.4
            ),
            tf.keras.layers.Dense(
                ntarget, 
                activation = 'softmax'
            ),
        ])
        print(self.clf.summary())
        self.clf.compile(
            loss = 'categorical_crossentropy',
            opimizer = 'adam',
            metrics = ['accuracy'],
        )
    
    def fit(self, X, y):
        print('Preprocessing')
        self.preprocessing.fit(X,y)
        print('Preprocessing end')
        self.clf.fit(
            self.preprocessing.transform(X), 
            y, 
            epochs = self.epochs,
            validation_split = 0.1
        )

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

    df = real_xml('./iniciativas08/', nfiles = None)
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
        ntarget = y_train.shape[1]
    )

    print(time.strftime("%X"))
    clf.fit(X = X_train, y = y_train)
    print(time.strftime("%X"))
    
if __name__ == "__main__":
    main()