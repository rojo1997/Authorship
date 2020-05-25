import unittest
import time
import os

from joblib import dump,load

import sys
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
    GRUClassifier,
    LSTMClassifier,
    SC1DClassifier,
    C1DSingleClassifier,
    C1DMultiClassifier
)

from numpy.random import seed
seed(1)
tf.random.set_seed(1)


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
        

    def test_TfidfVectorizer_ANOVA_MLPClassifier(self):
        X_train_preprocessing = load('data/gram_preprocessing_train.gz')
        X_test_preprocessing = load('data/gram_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)
        
        clf = MLPClassifier(
            input_shape = (30000,),
            num_classes = num_classes,
            dropout_rate = 0.2,
            regularization = 1e-4,
            units = [1024 * 2],
            layers = 1,
            verbose = True
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 50,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])
        
        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
        return(True)
    
    def test_TfidfVectorizer_ANOVA_LinearSVC(self):
        X_train_preprocessing = load('data/gram_preprocessing_train.gz')
        X_test_preprocessing = load('data/gram_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = LinearSVC(
            C = 1,
            penalty = 'l2',
            dual = True,
            loss = 'squared_hinge',
            intercept_scaling = 1,
            max_iter = 10000,
            random_state = self.random_state
        )

        clf.fit(X_train_preprocessing, y_train_encoder)
        print(clf.score(X_train_preprocessing, y_train_encoder))
        print(clf.score(X_test_preprocessing, y_test_encoder))
        y_test_pred = clf.predict(X_test_preprocessing)
        print(classification_report(
            encoder.inverse_transform(y_test_encoder),
            encoder.inverse_transform(y_test_pred),
        ))
        return(True)


    def test_TfidfVectorizer_ANOVA_LSA_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data(name, change_stdout = False)

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('Stemmer', Stemmer(language = 'spanish')),
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
            )),
            ('LSA', LSA(
                n_components = 700,
                n_iter = 15
            )),
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)

        clf = LinearSVC(
            C = 1.0,
            penalty = 'l2',
            dual = True,
            loss = 'squared_hinge',
            max_iter = 5000,
        )

        clf.fit(X_train_preprocessing, y_train_encoder)
        
        print(clf.score(X_train_preprocessing, y_train_encoder))

        X_test_preprocessing = preprocessing.transform(X_test)

        print(clf.score(X_test_preprocessing, y_test_encoder))
        return(True)

    def test_TfidfVectorizer_ANOVA_NMF_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data(name, change_stdout = False)

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('Stemmer', Stemmer(language = 'spanish')),
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
            )),
            ('NMF', NMF(
                n_components = 700
            )),
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)

        clf = LinearSVC(
            C = 1.0,
            penalty = 'l2',
            dual = True,
            loss = 'squared_hinge',
            max_iter = 5000,
        )

        clf.fit(X_train_preprocessing, y_train_encoder)
        
        print(clf.score(X_train_preprocessing, y_train_encoder))

        X_test_preprocessing = preprocessing.transform(X_test)

        print(clf.score(X_test_preprocessing, y_test_encoder))
        return(True)


    def test_StopWords_Stemmer_SC1DClassifier(self):
        X_train_preprocessing = load('data/sequences_preprocessing_train.gz')
        X_test_preprocessing = load('data/sequences_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = SC1DClassifier(
            input_shape = (512,),
            num_classes = num_classes,
            num_features = 30000,
            embedding_dim = 512,
            embedding_trainable = False,
            filters = 512,
            dropout_rate = 0.3,
            verbose = True,
            regularization = 1e-4 * 1,
            layers = 1
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 125,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])

        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
        
        clf.save('models/test_StopWords_Stemmer_SC1DClassifier.h5')
        return(True)

    def test_StopWords_Stemmer_C1DSingleClassifier(self):
        X_train_preprocessing = load('data/sequences_preprocessing_train.gz')
        X_test_preprocessing = load('data/sequences_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = C1DSingleClassifier(
            input_shape = (512,),
            num_classes = num_classes,
            num_features = 30000,
            embedding_dim = 1024 + 512,
            filters = 512,
            kernel_size = 3,
            regularization = 1e-4,
            dropout_rate = 0.2,
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 50,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])
        
        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
        return(True)

    def test_StopWords_Stemmer_C1DMultiClassifier(self):
        X_train_preprocessing = load('data/sequences_preprocessing_train.gz')
        X_test_preprocessing = load('data/sequences_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = C1DMultiClassifier(
            input_shape = (512,),
            num_classes = num_classes,
            num_features = 30000,
            embedding_dim = 1024,
            fkl = [
                (64,    1,  1),
                (512,   3,  1),
                (16,    5,  1)
            ],
            regularization = 1e-5,
            dropout_rate = 0.2,
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 50,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])
        
        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
        return(True)
    

    def test_StopWords_Stemmer_Sequences_LSTMClassifier(self):
        X_train_preprocessing = load('data/sequences_preprocessing_train.gz')
        X_test_preprocessing = load('data/sequences_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = LSTMClassifier(
            input_shape = (128,),
            num_classes = num_classes,
            num_features = 30000
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 50,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])
        
        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
        return(True)

    def test_StopWords_Stemmer_Sequences_GRUClassifier(self):
        X_train_preprocessing = load('data/sequences_preprocessing_train.gz')
        X_test_preprocessing = load('data/sequences_preprocessing_test.gz')
        y_train_encoder = load('data/y_train_encoder.gz')
        y_test_encoder = load('data/y_test_encoder.gz')
        num_classes = load('data/num_classes.gz')
        encoder = load('data/encoder.gz')
        print(X_train_preprocessing.shape)

        clf = GRUClassifier(
            input_shape = (512,),
            num_classes = num_classes,
            num_features = 30000
        )

        clf.fit(
            X_train_preprocessing,
            y_train_encoder,
            batch_size = 128,
            epochs = 50,
            validation_split = 0.1,
            callbacks = [EarlyStopping(patience = 3)],
            verbose = 2
        )

        print(clf.evaluate(
            X_train_preprocessing, 
            y_train_encoder, 
            batch_size = 128, 
            verbose = False
        )[1])

        print(clf.evaluate(
            X_test_preprocessing, 
            y_test_encoder, 
            batch_size = 128,
            verbose = False
        )[1])
        
        y_test_pred = encoder.inverse_transform(
            np.argmax(clf.predict(X_test_preprocessing), axis = 1)
        )
        print(classification_report(encoder.inverse_transform(y_test_encoder), y_test_pred))
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
    

class NeuralTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NeuralTest, self).__init__(*args, **kwargs)
        self.batch_size = 128
        self.epochs = 2
        self.n = 2000
        self.m = 50
        self.num_classes = 5
        self.num_features = 5000
    
    def test_MLPClassifier(self):
        x,y = self.generate_data(sparse = False)

        clf = MLPClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )

        x,y = self.generate_data(sparse = True)

        clf = MLPClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        return(True)

    def test_GRUClassifier(self):
        x,y = self.generate_data(sparse = False, integer = True)

        clf = GRUClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes,
            num_features = self.num_features
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        
        return(True)

    def test_LSTMClassifier(self):
        x,y = self.generate_data(sparse = False, integer = True)

        clf = LSTMClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes,
            num_features = self.num_features
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        
        return(True)

    def test_SC1DClassifier(self):
        x,y = self.generate_data(sparse = False, integer = True)

        clf = C1DSingleClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes,
            num_features = self.num_features
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        
        return(True)

    def test_C1DSingleClassifier(self):
        x,y = self.generate_data(sparse = False, integer = True)

        clf = C1DSingleClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes,
            num_features = self.num_features
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        
        return(True)

    def test_C1DMultiClassifier(self):
        x,y = self.generate_data(sparse = False, integer = True)

        clf = C1DMultiClassifier(
            input_shape = (self.m,),
            num_classes = self.num_classes,
            num_features = self.num_features
        )

        clf.fit(
            x = x,
            y = y,
            batch_size = 128,
            epochs = 5,
            verbose = 2
        )
        
        return(True)

    def generate_data(self, sparse = False, integer = False):
        import scipy
        import numpy as np

        x = None
        if integer: 
            x = np.random.randint(
                low = 0,
                high = self.num_features,
                size = (self.n,self.m)
            )
        else: 
            x = np.random.rand(self.n,self.m)
        if sparse: 
            x = scipy.sparse.csr_matrix(x)
        y = np.random.randint(low = 0, high = self.num_classes, size = self.n)

        return(x,y)


class FeatureExtraction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FeatureExtraction, self).__init__(*args, **kwargs)

        # Parametros genericos para cualquier test
        self.random_state = 1
        self.nwords = 50
        self.frecuency = 200
        self.subset = None
        self.nfiles = None
        self.sample = 10000
        self.test_size = 0.2
        self.verbose = True

    def test_TfidfVectorizer_ANOVA(self):
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data()

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('TfidfVectorizer', TfidfVectorizer(
                max_features = 100000,
                ngram_range = (1,4),
                analyzer = 'word',
                encoding = 'utf8',
                dtype = np.float32,
                min_df =   1 / 1000.0,
                max_df = 9999.0 /  1000.0,
                strip_accents = None,
                decode_error = 'replace',
                lowercase = False
            )),
            ('ANOVA', ANOVA(
                k = 30000
            ))
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)
        X_test_preprocessing = preprocessing.transform(X_test)

        dump(X_train_preprocessing, 'data/gram_preprocessing_train.gz')
        dump(X_test_preprocessing, 'data/gram_preprocessing_test.gz')
        dump(y_train_encoder, 'data/y_train_encoder.gz')
        dump(y_test_encoder, 'data/y_test_encoder.gz')
        dump(num_classes, 'data/num_classes.gz')
        dump(encoder, 'data/encoder.gz')
        dump(preprocessing, 'models/gram_preprocessing.gz')

    def test_TfidfVectorizer(self):
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data()

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            ('Stemmer', Stemmer(language = 'spanish')),
            ('TfidfVectorizer', TfidfVectorizer(
                max_features = 100000,
                ngram_range = (2,3),
                analyzer = 'word',
                encoding = 'utf8',
                dtype = np.float32,
                min_df =   1.0 / 10000.0,
                max_df = 999.0 /  1000.0,
                strip_accents = None,
                decode_error = 'replace',
                lowercase = True
            ))
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)
        print(X_train_preprocessing.shape)

    def test_Sequences(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes, encoder = self.read_data()

        y_train_encoder = encoder.transform(y_train)
        y_test_encoder = encoder.transform(y_test)

        preprocessing = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('Sequences', Sequences(
                num_words = 30000, 
                maxlen = 512, 
                padding = 'post', 
                truncating = 'post'
            ))
        ], verbose = True)

        X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)
        X_test_preprocessing = preprocessing.transform(X_test)

        dump(X_train_preprocessing, 'data/sequences_preprocessing_train.gz')
        dump(X_test_preprocessing, 'data/sequences_preprocessing_test.gz')
        dump(y_train_encoder, 'data/y_train_encoder.gz')
        dump(y_test_encoder, 'data/y_test_encoder.gz')
        dump(num_classes, 'data/num_classes.gz')
        dump(encoder, 'data/encoder.gz')
        dump(preprocessing, 'models/sequences_preprocessing.gz')


    def read_data(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
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


class PreprocessingTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PreprocessingTest, self).__init__(*args, **kwargs)

    def test_Translate(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio.'])

        clf = Pipeline(steps = [
            ('Translate', Translate(src = 'es', dest = 'en')),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_Stemmer(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio.'])

        clf = Pipeline(steps = [
            ('Stemmer', Stemmer(language = 'spanish')),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_Puntuation(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio.'])

        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_StopWords(self):
        X_train = pd.Series(['La casa del barrio esta en mi municipio.'])

        clf = Pipeline(steps = [
            ('StopWords', StopWords(language = 'spanish'))
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)
    
    def test_Lemmatizer(self):
        X_train = pd.Series(['It gives us the measure of how far the predictions were from the actual output.'])

        clf = Pipeline(steps = [
            ('Lemmatizer', Lemmatizer()),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)

    def test_PosTag(self):
        X_train = pd.Series(['It gives us the measure of how far the predictions were from the actual output.'])

        clf = Pipeline(steps = [
            ('PosTag', PosTag()),
        ], verbose = True)
        result = clf.fit_transform(X_train)
        print(result)
        
        return(True)


class DataExtractionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DataExtractionTest, self).__init__(*args, **kwargs)
        self.nwords = 50
        self.frecuency = 200

    def test_real_xml(self):
        df = real_xml('./iniciativas08/')
    
    def test_filter_df(self):
        df = real_xml('./iniciativas08/')
        df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)

    def test_clean(self):
        frase = '  ¿Esto es una    prueba de  [pausa] limpieza (3.0)  ?!! Sería        buena'
        print(clean(frase))

if __name__ == "__main__":
    unittest.main(verbosity = 2)