###############################################################################


###############################################################################

import numpy as np
import pandas as pd
from joblib import Memory

from shutil import rmtree

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer

from nltk.stem.snowball import SpanishStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from string import punctuation

from NVarPrint import NVarPrint

class Authorship(BaseEstimator, ClassifierMixin):
    def __init__(self, le, random_state = 1, verbose = False):
        # Extrator de raices en castellano
        self.stemmer = SpanishStemmer()
        # Analizador externo
        self.analyzer = CountVectorizer().build_analyzer()
        # Codificacion para los nombres en Y
        self.le = le
        # Memoria de cache adicional para el pipeline
        self.location = 'cachedir'
        self.memory = Memory(location = self.location, verbose = 10)

        self.text_clf = Pipeline([
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
                dtype = np.int64
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
            ('NVarPrint', NVarPrint(
                # Imprimir datos estadísticos
                verbose = True if verbose == 2 else False
            )),
            ('LinearSVC', LinearSVC(
                penalty = 'l2', 
                loss = 'squared_hinge', 
                dual = True,
                # Tipo de resulucion de problema multiclase: one vs rest
                multi_class = 'ovr',
                fit_intercept = True,
                intercept_scaling = 1,
                # Peso de las clases idéntico
                class_weight = None,
                # Fijar el factor aleatorio
                random_state = random_state,
                # Maximo de iteraciones
                max_iter = 5000,
                # Verbose
                verbose = True if verbose == 3 else False
            ))
        ], verbose = True if verbose == 2 else False)

        self.param_grid = {
            # Tomar palabras en duos, trios...
            'CountVectorizer__ngram_range': [(1,1), (1,2), (1,3), (2,3)],
            # Numero de caracteristicas extraidas del texto
            'CountVectorizer__max_features': [1000, 5000, None],
            # Frecuencia maxima en el texto
            'CountVectorizer__max_df': [1.0, 0.9, 0.8, 0.7],
            # Frecuencia minima en el texto
            #'min_df': [1e-1, 1e-2],
            # Toleracia a parada
            'LinearSVC__tol': [1e-3, 1e-4, 1e-5],
            # Penalizacion por termino mal clasificado
            'LinearSVC__C': [2.0, 1.0, 0.5]
        }

        self.scoring = make_scorer(
            # Funcion de score
            score_func = accuracy_score,
            # Cuanto mayor sea el valor mejor
            greater_is_better = True
        )

        self.clf = GridSearchCV(
            # Clasificar para optimizar hiperparametros
            estimator = self.text_clf,
            # Espacio de hiperparametros a explorar
            param_grid = self.param_grid,
            # Medicion de error
            scoring = self.scoring,
            # Numero de hebras
            n_jobs = 4,
            # Numero de validaciones
            cv = 5,
            # Imprimir progreso
            verbose = 7 if verbose == 1 else False
        )
    
    def fit(self, X, y):
        self.clf.fit(X = X, y = self.le.transform(y))

    def predict(self, X):
        return(self.clf.predict(X))

    def score(self, X, y):
        return(self.clf.score(X,self.le.transform(y)))

    