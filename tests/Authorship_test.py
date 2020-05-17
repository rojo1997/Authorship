import unittest
import time
from dill import load, dump
import matplotlib.pyplot as plt
import sys
sys.path[0] = sys.path[0].replace('\\tests','')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from nltk.corpus import stopwords
from string import punctuation

import pandas as pd
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

        # Salida de datos
        sys.stderr = open('results/stderr.txt', encoding = 'utf-8', mode = 'a')
        self.stdout = open('results/stdout.txt', encoding = 'utf-8', mode = 'a')
        """try:
            self.stdout = open('results/report.txt', encoding = 'utf-8', mode = 'a')
        except:
            pass
        else:
            self.stdout = open('results/report.txt', encoding = 'utf-8', mode = 'w')"""

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
            'batch_size': 256,
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
        stdout = sys.stdout
        sys.stdout = open('results/' + sys._getframe().f_code.co_name + '.txt', encoding = 'utf-8', mode = 'w+')
        
        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        print('datos xml: ', df.shape)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        print('datos despues del filtro: ', df.shape)

        print('='.join(['' for n in range(80)]))
        sys.stdout = stdout

    def test_translate(self):
        stdout = sys.stdout
        sys.stdout = open('results/' + sys._getframe().f_code.co_name + '.txt', encoding = 'utf-8', mode = 'w+')

        from googletrans import Translator
        translator = Translator()
        print(translator.translate('hola.', src = 'es'))

        print('='.join(['' for n in range(80)]))
        sys.stdout = stdout

    def test_words(self):
        stdout = open('results/' + sys._getframe().f_code.co_name + '.txt', encoding = 'utf-8', mode = 'w+')

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

        print('='.join(['' for n in range(80)]))
        sys.stdout = stdout
    
    def test_stop(self):
        stdout = open('results/' + sys._getframe().f_code.co_name + '.txt', encoding = 'utf-8', mode = 'w+')

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

        print('='.join(['' for n in range(80)]))
        sys.stdout = stdout

    def test_clean(self):
        stdout = open('results/' + sys._getframe().f_code.co_name + '.txt', encoding = 'utf-8', mode = 'w+')

        frase = '  ¿Esto es una    prueba de  [pausa] limpieza (3.0)  ?!! Sería        buena'
        print(clean(frase))
        for w in stopwords.words("spanish"):
            print(w, sep = '', end = ', ')
        #print(stopwords.words("spanish"))

        print('='.join(['' for n in range(80)]))
        sys.stdout = stdout

    def test_TfidfVectorizer_ANOVA_MLPClassifier(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        model = KerasClassifier(
            MLPClassifier,
            num_classes = num_classes,
            **self.MLPClassifier_params
        )
        param_grid = {
            'layers': [1,2],
            'units': [32,64,128,128 + 16],
            'dropout_rate': [0.1,0.2,0.3,0.4]
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
                cv = 5
            ))
        ], verbose = True)

        clf.fit(X_train, y_train)
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)
    
    def test_TfidfVectorizer_ANOVA_LinearSVC(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

        model = LinearSVC(
            **self.LinearSVC_params
        )
        param_grid = {
            'C': [0.1,0.5,1.0,1.5,2,2.5]
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
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
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
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

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
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def test_StopWords_Stemmer_Sequences_SC1D(self):
        name = sys._getframe().f_code.co_name
        X_train, X_test, y_train, y_test, num_classes = self.open(name)

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
        
        self.close(name, clf, param_grid, X_train, X_test, y_train, y_test)
        return(True)

    def open(self, name):
        self.stdout = sys.stdout
        sys.stdout = open('results/' + name + '.txt', encoding = 'utf-8', mode = 'w+')

        df = real_xml('./iniciativas08/', nfiles = self.nfiles)
        if self.subset != None: df = df.sample(self.sample, random_state = self.random_state)
        if self.nfiles == None: df = filter_df(df, nwords = self.nwords, frecuency = self.frecuency)
        df = df.head(100)
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

        return(X_train, X_test, y_train, y_test, num_classes)

    def close(self, name, clf, param_grid, X_train, X_test, y_train, y_test):
        print(clf['GridSearchCV'].best_params_)
        print(clf['GridSearchCV'].cv_results_)
        print(clf['GridSearchCV'].cv)
        print("Accuracy train: ", clf.score(X = X_train, y = y_train))
        print("Accuracy test: ", clf.score(X = X_test, y = y_test))
        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred))
        print(confusion_matrix(y_test, y_test_pred))

        self.graph(
            name, 
            clf['GridSearchCV'].best_index_, 
            clf['GridSearchCV'].cv_results_,
            clf['GridSearchCV'].best_params_,
            param_grid
        )

        if isinstance(clf['GridSearchCV'].best_estimator_, tf.keras.wrappers.scikit_learn.KerasClassifier):
            clf['GridSearchCV'].best_estimator_.model.save('models/model_' + name + '.h5')
            clf['GridSearchCV'].best_estimator_.model = None
        file_out = open('models/model_' + name + '.pkl', mode = 'wb+')
        dump(clf, file_out)
        file_out.close()

        print('='.join(['' for n in range(80)]))
        sys.stdout = self.stdout

    def graph(self, name, best_index_, cv_results_, best_params_, param_grid):
        X = [
            'Div 0',
            'Div 1',
            'Div 2',
            'Div 3',
            'Div 4',
        ]
        Y = [
            cv_results_['split0_test_score'][best_index_],
            cv_results_['split1_test_score'][best_index_],
            cv_results_['split2_test_score'][best_index_],
            cv_results_['split3_test_score'][best_index_],
            cv_results_['split4_test_score'][best_index_],
        ]
        fig, ax = plt.subplots()
        ax.bar(X,Y)
        plt.xlabel('Divisiones')
        plt.ylabel('Puntuación')
        plt.title('Mejor puntuación: ' + str(best_params_))
        fig.savefig(
            'images/' + name + 'best_score_splits',
            dpi = 400
        )
        fig.clf()

        for param, values in param_grid.items():
            if len(values) == 1: continue
            param_rest = [*param_grid]
            param_rest.remove(param)
            index, Y = map(list,zip(*[
                (cv_results_['params'].index(ps),ps[param]) 
                for ps in cv_results_['params'] 
                if all([best_params_[p] == ps[p] for p in param_rest])
            ]))
            df = pd.DataFrame(columns = ['values','mean', 'std'])
            df['values'] = Y
            df['mean'] = cv_results_['mean_test_score'][index]
            df['std'] = cv_results_['std_test_score'][index]
            df.fillna(value = 0.0, inplace = True)
            if df['values'].dtype == 'float64' or df['values'].dtype == 'int64':
                fig, ax = plt.subplots()
                ax.set(ylim=(0.0, 1.0))
                ax.fill_between(
                    df['values'].values, 
                    df['mean'] - df['std'],
                    df['mean'] + df['std'],
                    color = 'blue',
                    alpha = 0.1
                )
                ax.plot(
                    df['values'].values,
                    df['mean'].values,
                    color = 'black'
                )
                plt.xlabel(param)
                plt.ylabel('Puntuación media')
                plt.title('Puntuación media: ' + param + ' ' + '' if len(param_rest) == 0 else str([p + '=' + str(best_params_[p]) for p in param_rest]))
                
                fig.savefig(
                    'images/' + name + '_' + param,
                    dpi = 400
                )
                fig.clf()
            else:
                fig, ax = plt.subplots()
                ax.bar(df['values'].values,df['mean'].values)
                plt.xlabel(param)
                plt.ylabel('Puntuación media')
                plt.title('Puntuación media: ' + param + ' ' + str([p + '=' + str(best_params_[p]) for p in param_rest]))
                fig.savefig(
                    'images/' + name + '_' + param,
                    dpi = 400
                )
                fig.clf()

            

if __name__ == "__main__":
    unittest.main(verbosity = 2)