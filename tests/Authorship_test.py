import unittest
import time
import sys
import os

sys.path[0] = sys.path[0].replace('\\tests','')

from tests.MLPlatform import MLPlatform

from nltk.corpus import stopwords
from string import punctuation

import pandas as pd
import numpy as np

import tensorflow as tf

from keras.callbacks import EarlyStopping

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV
)

from sklearn.metrics import (
    classification_report, 
    confusion_matrix
)

from Authorship.functions import (
    real_xml, 
    filter_df, 
    clean
)

from Authorship.feature_selection import ANOVA

from sklearn.svm import LinearSVC

from sklearn.decomposition import (
    TruncatedSVD as LSA,
    NMF
)

from Authorship.preprocessing import (
    StopWords, 
    Stemmer, 
    Lemmatizer,
    Puntuation, 
    Translate,
    PosTag
)

from Authorship.feature_extration.text import Sequences
from sklearn.feature_extraction.text import TfidfVectorizer

from Authorship.neural_network import (
    MLPClassifier,
    LSTMClassifier, 
    GRUClassifier, 
    SC1DClassifier,
    Conv1D_SingleKernel,
    Conv1D_MultiKernel
)


class AuthorshipTest(unittest.TestCase, MLPlatform):
    def __init__(self, *args, **kwargs):
        super(AuthorshipTest, self).__init__(*args, **kwargs)

        # Parametros genericos para cualquier test
        self.random_state = 1
        self.nwords = 50
        self.frecuency = 200
        self.subset = None
        self.nfiles = None
        self.sample = 10000
        self.test_size = 0.2
        self.verbose = True

        # Salida de datos
        

        # Parametros TfidfVectorizer
        self.TfidfVectorizer_params = {
            'max_features': 100000,
            'ngram_range': (1,2),
            'analyzer': 'word',
            'encoding': 'utf8',
            'dtype': np.float32,
            'min_df': 1.0 / 1000.0,
            'max_df': 999.0 / 1000.0,
            'strip_accents': None,
            'decode_error': 'replace',
            'lowercase': True
        }

        # Parametros ANOVA
        self.ANOVA_params = {
            'k': 20000
        }

        # Parametros MLPClassifier
        self.MLPClassifier_params = {
            'layers': 1,
            'epochs': 30,
            'input_shape': (self.ANOVA_params['k'],),
            'verbose': 2
        }

        # Parametros LinearSVC
        self.LinearSVC_params = {
            'C': 1.0,
            'penalty': 'l2',
            'dual': True,
            'loss': 'squared_hinge',
            'max_iter': 5000,
        }

        # Parametros LSA
        self.LSA_params = {
            'n_components': 700,
            'n_iter': 15
        }

        # Parametros MLPClassifier
        self.MLPClassifier_params_2 = {
            'layers': 1,
            'units': 20,
            'dropout_rate': 0.3,
            'epochs': 150,
            'batch_size': 1024,
            'input_shape': self.LSA_params['n_components'],
            'sparse': False,
            'verbose': 2
        }

        # Parametros NMF
        self.NMF_params = {
            'n_components': 700
        }

        # Parametros MLPClassifier
        self.MLPClassifier_params_3 = {
            'layers': 1,
            'units': 96,
            'dropout_rate': 0.3,
            'epochs': 100,
            'batch_size': 1024,
            'input_shape': self.LSA_params['n_components'],
            'sparse': False,
            'verbose': False
        }

        

        # Parametros Stemmer
        self.Stemmer_params = {
            'language': 'spanish'
        }

        # Parametros Sequences
        self.Sequences_params = {
            'num_words': 10000, 
            'maxlen': 256, 
            'padding': 'post', 
            'truncating': 'post'
        }
        """self.Sequences_params = {
            'num_words': 60000, 
            'maxlen': 512 + 128, 
            'padding': 'post', 
            'truncating': 'post'
        }"""

        # Parametros LSTMClassifier
        self.LSTMClassifier_params = {
            'embedding_dim': 256,
            'dropout_rate': 0.3,
            'epochs': 40,
            'input_shape': (self.Sequences_params['maxlen'],),
            'num_features': self.Sequences_params['num_words'],
        }

        # Parametros GRUClassifier
        self.GRUClassifier_params = {
            'layers': 1,
            'embedding_dim': 175,
            'dropout_rate': 0.3,
            'epochs': 15,
            'input_shape': (self.Sequences_params['maxlen'],),
            'num_features': self.Sequences_params['num_words'],
        }

        # Parametros SC1DClassifier
        self.SC1DClassifier_params = {
            'layers': 1,
            'embedding_dim': 128,
            'filters': 64,
            'kernel_size': 3,
            'regularize': 1e-5,
            'dropout_rate': 0.1,
            'epochs': 80,
            'input_shape': (self.Sequences_params['maxlen'],),
            'num_features': self.Sequences_params['num_words'],
        }

    def test_read_xml_filter(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        print('='.join(['' for n in range(80)]))
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout

    def test_translate(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio'])

        clf = Pipeline(steps = [
            ('Translate', Translate(src = 'es', dest = 'en')),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_stemmer(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio'])

        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            ('Stemmer', Stemmer(language = 'spanish')),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)
    
    def test_lemmatizer(self):
        X_train = pd.Series(['It gives us the measure of how far the predictions were from the actual output.'])

        clf = Pipeline(steps = [
            ('Lemmatizer', Lemmatizer()),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)

        return(True)

    def test_pos_tag(self):
        X_train = pd.Series(['It gives us the measure of how far the predictions were from the actual output.'])

        clf = Pipeline(steps = [
            ('PosTag', PosTag()),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_stop(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio'])

        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish'))
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_clean(self):
        frase = '  ¿Esto es una    prueba de  [pausa] limpieza (3.0)  ?!! Sería        buena'
        print(clean(frase))

    def test_TfidfVectorizer_ANOVA_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data(name, change_stdout = False)

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            #('Stemmer', Stemmer(language = 'spanish')),
            ('TfidfVectorizer', TfidfVectorizer(
                max_features = 100000,
                ngram_range = (1,2),
                analyzer = 'word',
                encoding = 'utf8',
                dtype = np.float32,
                min_df = 1.0 / 1000.0,
                max_df = 999.0 / 1000.0,
                strip_accents = None,
                decode_error = 'replace',
                lowercase = True
            )),
            ('ANOVA', ANOVA(
                k = 20000
            ))
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)
        
        clf = MLPClassifier(
            input_shape = (20000,),
            num_classes = num_classes,
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 40,
            validation_split = 0.1,
            verbose = 2
        )

        print(clf.summary())
        
        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        ))

        X_test_preprocessing = preprocessing.transform(X_test)

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        ))
        return(True)
    
    def test_TfidfVectorizer_ANOVA_LinearSVC(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.read_data(name)

        model = LinearSVC(
            **self.LinearSVC_params
        )
        param_grid = {
            'C': [0.5,1.0,5,10,25,50,75,100]
        }
        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('Stemmer', Stemmer(language = 'spanish')),
            ('TfidfVectorizer', TfidfVectorizer(
                **self.TfidfVectorizer_params
            )),
            ('ANOVA', ANOVA(
                **self.ANOVA_params
            )),
            ('GridSearchCV', GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                verbose = self.verbose,
                n_jobs = 4
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.gridsearchcv_graph(
            name = name, 
            gridsearchcv = clf['GridSearchCV']
        )
        self.generate_report(
            name = name,
            clf = clf,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            gridsearchcv = clf['GridSearchCV']
        )
        self.dump_model(
            name = name,
            clf = clf, 
            keras_model = None
        )
        self.close()
        return(True)

    def test_TfidfVectorizer_ANOVA_LSA_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params_2
        )
        param_grid = {
            'layers': [1,2],
            'units': [32,64,96],
            'dropout_rate': [0.1,0.2,0.3,0.4]
        }
        clf = Pipeline(steps = [
            ('TfidfVectorizer', TfidfVectorizer(
                **self.TfidfVectorizer_params
            )),
            ('ANOVA', ANOVA(
                **self.ANOVA_params
            )),
            ('LSA', LSA(
                **self.LSA_params
            )),
            ('GridSearchCV', GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                verbose = self.verbose
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def test_TfidfVectorizer_ANOVA_NMF_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params_3
        )
        param_grid = {
            'layers': [1,2],
            'units': [32,64,96,128],
            'dropout_rate': [0.1,0.2,0.3,0.4]
        }
        clf = Pipeline(steps = [
            ('TfidfVectorizer', TfidfVectorizer(
                **self.TfidfVectorizer_params
            )),
            ('ANOVA', ANOVA(
                **self.ANOVA_params
            )),
            ('NMF', NMF(
                **self.NMF_params
            )),
            ('GridSearchCV', GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                verbose = self.verbose
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def test_StopWords_Stemmer_Sequences_LSTM(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        param_grid = {
            'embedding_dim': [100],
            'dropout_rate': [0.2]
        }

        model = KerasClassifier(
            LSTMClassifier,
            num_classes = num_classes,
            **self.LSTMClassifier_params,
            verbose = 2,
            validation_split = 0.1,
            batch_size = 64 * 2
        )

        """callback = tf.keras.callbacks.EarlyStopping(
            monitor = 'loss', 
            patience = 3
        )

        model.set_params(callbacks = [callback])"""

        clf = Pipeline(steps = [
            #('Translate', Translate(src = 'es', dest = 'en')),
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            #('Stemmer', Stemmer(language = 'spanish')),
            ('Sequences', Sequences(
                **self.Sequences_params
            )),
            ('LSTMClassifier', KerasClassifier(
                LSTMClassifier,
                num_classes = num_classes,
                **self.LSTMClassifier_params,
                verbose = 2,
                validation_split = 0.1,
                batch_size = 64 * 2
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def test_StopWords_Stemmer_Sequences_GRU(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name, change_stdout = False)

        param_grid = {
            'layers': [2],
            'units': [64],
            'dropout_rate': [0.2],
            'regularize': [1e-4]
        }

        model = KerasClassifier(
            GRUClassifier,
            num_classes = num_classes,
            **self.GRUClassifier_params,
            verbose = 2,
            batch_size = 64,
            validation_split = 0.1
        )

        clf = Pipeline(steps = [
            #('Translate', Translate(src = 'es', dest = 'en')),
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            #('Stemmer', Stemmer(language = 'spanish')),
            ('Sequences', Sequences(
                **self.Sequences_params
            )),
            ('GridSearchCV', GridSearchCV(
                model,
                param_grid = param_grid,
                cv = 5,
                n_jobs = 1,
                verbose = True
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def test_StopWords_Stemmer_Sequences_SC1D(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.read_data(name)

        param_grid = {
            'layers': [3],
            'filters': [128],
            'dropout_rate': [0.2],
            'regularize': [1e-5],
            'kernel_size': [5]
        }
        # 1, 75, 0.1: 27

        model = KerasClassifier(
            SC1DClassifier,
            num_classes = num_classes,
            **self.SC1DClassifier_params,
            verbose = 2,
            batch_size = 64,
            validation_split = 0.1
        )

        clf = Pipeline(steps = [
            #('Translate', Translate(src = 'es', dest = 'en')),
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            #('Stemmer', Stemmer(language = 'spanish')),
            ('Sequences', Sequences(
                **self.Sequences_params
            )),
            ('GridSearchCV', GridSearchCV(
                model,
                param_grid = param_grid,
                cv = 5,
                n_jobs = 1,
                verbose = 9
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)

        self.gridsearchcv_graph(
            name = name, 
            gridsearchcv = clf['GridSearchCV']
        )
        self.generate_report(
            name = name,
            clf = clf,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            gridsearchcv = clf['GridSearchCV']
        )
        self.dump_model(
            name = name,
            clf = clf, 
            keras_model = None
        )
        self.close()
        return(True)
    
    def test_StopWords_Stemmer_Sequences_SC1D_m(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data(name, change_stdout = False)

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        maxlen = 512 + 128
        num_words = 60000

        preprocessing = Pipeline(steps = [
            #('Translate', Translate(src = 'es', dest = 'en')),
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            #('Stemmer', Stemmer(language = 'spanish')),
            ('Sequences', Sequences(
                num_words = num_words, 
                maxlen = maxlen, 
                padding = 'post', 
                truncating = 'post'
            ))
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)

        """clf = Embedding_Conv1D_GlobalMaxPooling1D_Dense(
            layers = 1,
            embedding_dim = 1024 + 512,
            filters = 512,
            kernel_size = 3,
            regularization = 1e-4,
            dropout_rate = 0.2,
            input_shape = (maxlen,),
            num_classes = num_classes,
            num_features = num_words
        )"""
        clf = Conv1D_MultiKernel(
            embedding_dim = 1024,
            fkl = [
                (64,    1,  1),
                (512,   3,  1),
                (16,    5,  1)
            ],
            regularization = 1e-5,
            dropout_rate = 0.2,
            input_shape = (maxlen,),
            num_classes = num_classes,
            num_features = num_words
        )
        print(clf.summary())

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 40,
            validation_split = 0.1,
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        ))

        X_test_preprocessing = preprocessing.transform(X_test)

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        ))
        return(True)

    def read_data(self, name, change_stdout = True):
        super(AuthorshipTest, self).read_data(name, change_stdout = change_stdout)

        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        #df = df.head(3000)
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )
        print('train', X_train.shape, y_train.shape)
        print('test', X_test.shape, y_test.shape)

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        encoder = LabelEncoder()
        encoder.fit(df['name'])

        return(X_train, X_test, y_train, y_test, num_classes, encoder)

    def test_load_model(self):
        name = 'test_TfidfVectorizer_ANOVA_LinearSVC'
        name = 'test_TfidfVectorizer_ANOVA_MLPClassifier'
        clf = self.load_model(name)
        while True:
            frase = 'hola que tal'
            print(clf.predict(pd.Series([frase], name = 'text')))
    


if __name__ == "__main__":
    unittest.main(verbosity = 2)