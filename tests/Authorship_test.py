import unittest
import time

import sys
sys.path[0] = sys.path[0].replace('\\tests','')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from nltk.corpus import stopwords
from string import punctuation

import numpy as np

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# Extraccion y filtrado de datos
from Authorship.functions import real_xml, filter_df, clean

# MODELO: TfidfVectorizer, ANOVA, MLPClasiffier
from sklearn.feature_extraction.text import TfidfVectorizer
from Authorship.feature_selection import ANOVA
from Authorship.neural_network import MLPClassifier

# MODELO: test_TfidfVectorizer_ANOVA_SVCLinear
#from sklearn.feature_extraction.text import TfidfVectorizer
#from Authorship.feature_selection import ANOVA
from sklearn.svm import LinearSVC

# MODELO: test_TfidfVectorizer_ANOVA_LSA_MLPClasiffier
#from sklearn.feature_extraction.text import TfidfVectorizer
#from Authorship.feature_selection import ANOVA
from sklearn.decomposition import TruncatedSVD as LSA
#from Authorship.neural_network import MLPClassifier

# MODELO: test_TfidfVectorizer_ANOVA_NMF_MLPClasiffier
#from sklearn.feature_extraction.text import TfidfVectorizer
#from Authorship.feature_selection import ANOVA
from sklearn.decomposition import NMF
#from Authorship.neural_network import MLPClassifier

from Authorship.preprocessing import StopWords, Stemmer, Puntuation, Translate
from Authorship.feature_extration.text import Sequences
from Authorship.neural_network import LSTMClassifier, GRUClassifier, SC1DClassifier

"""from Authorship.TFIDFANOVAMLP.Authorship import Authorship as MLP
from Authorship.TFIDFAVOVASVM.Authorship import Authorship as SVM
from Authorship.SESEPCNN.Authorship import Authorship as SEPCNN
from Authorship.SELSTM.Authorship import Authorship as LSTM"""


class AuthorshipTest(unittest.TestCase):
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

        # Parametros TfidfVectorizer
        self.TfidfVectorizer_params = {
            'stop_words': stopwords.words("spanish"),
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
            'units': 128,
            'dropout_rate': 0.3,
            'epochs': 100,
            'batch_size': 1024,
            'input_shape': self.ANOVA_params['k'],
            'sparse': True,
            'verbose': False
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
            'units': 128,
            'dropout_rate': 0.3,
            'epochs': 150,
            'batch_size': 1024,
            'input_shape': self.LSA_params['n_components'],
            'sparse': False,
            'verbose': False
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

        # Parametros StopWords
        self.StopWords_params = {
            'stop_words': stopwords.words("spanish")
        }

        # Parametros Stemmer
        self.Stemmer_params = {
            'language': 'spanish'
        }

        # Parametros Sequences
        self.Sequences_params = {
            'num_words': 15000, 
            'maxlen': 256, 
            'padding': 'post', 
            'truncating': 'post'
        }

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
            'epochs': 40,
            'input_shape': (self.Sequences_params['maxlen'],),
            'num_features': self.Sequences_params['num_words'],
        }

    def test_read_xml_filter(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        self.assertGreater(df.shape[0], 0)

    def test_translate(self):
        from googletrans import Translator
        translator = Translator()
        print(translator.translate('hola.', src = 'es'))

    def test_TfidfVectorizer_ANOVA_MLPClassifier(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )
        print(X_train)
        print(X_test)

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params
        )
        param_grid = {
            'layers': [1],
            'units': [128],
            'dropout_rate': [0.3]
        }
        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
            ('TfidfVectorizer', TfidfVectorizer(
                **self.TfidfVectorizer_params
            )),
            ('ANOVA', ANOVA(
                **self.ANOVA_params
            )),
            ('GridSearchCV', GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                verbose = self.verbose
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))
        return(True)
    
    def test_TfidfVectorizer_ANOVA_LinearSVC(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        model = LinearSVC(
            **self.LinearSVC_params
        )
        param_grid = {
            'C': [0.9,1.0,1.1]
        }
        clf = Pipeline(steps = [
            ('Puntuation', Puntuation()),
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
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))
        return(True)

    def test_TfidfVectorizer_ANOVA_LSA_MLPClassifier(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params_2
        )
        param_grid = {
            'layers': [1,2],
            'units': [128],
            'dropout_rate': [0.3]
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
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))
        return(True)

    def test_TfidfVectorizer_ANOVA_NMF_MLPClassifier(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params_3
        )
        param_grid = {
            'layers': [1,2],
            'units': [64,96],
            'dropout_rate': [0.2,0.3]
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
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))
        return(True)

    def test_StopWords_Stemmer_Sequences_LSTM(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        #df = df.head(20000)
        print(df['name'].value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

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

        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))

    def test_StopWords_Stemmer_Sequences_GRU(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        #df = df.head(1000)
        print(df['name'].value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        param_grid = {
            'layers': [1],
            'embedding_dim': [256 + 128],
            'dropout_rate': [0.1,0.2,0.3]
        }
        # 1, 75, 0.1: 27

        model = KerasClassifier(
            GRUClassifier,
            num_classes = num_classes,
            **self.GRUClassifier_params,
            verbose = 2,
            batch_size = 64 * 2,
            #validation_split = 0.1
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
            ('Stemmer', Stemmer(language = 'spanish')),
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
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))

    def test_StopWords_Stemmer_Sequences_SC1D(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        print(df['name'].value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        labels = list(np.unique(df['name']))
        num_classes = len(labels)

        param_grid = {
            'layers': [1],
            'filters': [64, 64 + 16],
            'dropout_rate': [0.1,0.2],
            'regularize': [1e-5]
        }
        # 1, 75, 0.1: 27

        model = KerasClassifier(
            SC1DClassifier,
            num_classes = num_classes,
            **self.SC1DClassifier_params,
            verbose = False,
            batch_size = 64,
            #validation_split = 0.1
        )

        #print(model.model.summary())

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
            ('GridSearchCV', GridSearchCV(
                model,
                param_grid = param_grid,
                cv = 5,
                n_jobs = 1,
                verbose = 9
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        print(clf['GridSearchCV'].cv_results_)
        print(clf['GridSearchCV'].cv)
        print(clf['GridSearchCV'].best_params_)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))

    def test_words(self):
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        
        print(df['name'].value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['name'],
            test_size = self.test_size,
            random_state = self.random_state
        )

        clf = Pipeline(steps = [
            #('Translate', Translate(src = 'es', dest = 'en')),
            ('Puntuation', Puntuation()),
            ('StopWords', StopWords(language = 'spanish')),
            ('Stemmer', Stemmer(language = 'spanish')),
        ], verbose = True)

        X_train['text'] = clf.fit_transform(X_train, y_train)
        print(X_train['text'])
        print(X_train['text'])
        X_train['nwords'] = X_train['text'].apply(lambda s: len(s.split(' ')))
        print(X_train['nwords'].describe())
    
    def test_stop(self):
        """print(self.TfidfVectorizer_params['stop_words'])"""
        print('sería' in self.TfidfVectorizer_params['stop_words'])
        
        #print('|'.join(map(lambda s: re.escape(' ' + s + ' '), self.TfidfVectorizer_params['stop_words'])))
        import re
        print('[' + '|'.join(["( |^)" + w + "( |^)" for w in self.TfidfVectorizer_params['stop_words']]) + ']*')
        filters = re.compile('[' + '|'.join(["( |^)" + w + "( |^)" for w in self.TfidfVectorizer_params['stop_words']]) + ']*')
        a = filters.sub(' ', 'la casa tiena la tendrá sería que Es soy '.lower())
        print(a)
        b = filters.sub(' ', a.lower())
        print(b)
        c = filters.sub(' ', b.lower())
        print(c)

    def test_clean(self):
        frase = '  ¿Esto es una    prueba de  [pausa] limpieza (3.0)  ?!! Sería        buena'
        print(clean(frase))
        for w in stopwords.words("spanish"):
            print(w, sep = '', end = ', ')
        #print(stopwords.words("spanish"))

if __name__ == "__main__":
    unittest.main(verbosity = 2)