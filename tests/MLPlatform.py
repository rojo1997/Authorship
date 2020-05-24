import time
from dill import load, dump
import joblib
import matplotlib.pyplot as plt
import sys
import os
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf


class MLPlatform():
    def __init__(self, *args, **kwargs):
        pass

    def iskeras(self, model):
        return(isinstance(model, KerasClassifier))

    def read_data(self, name, change_stdout = True):
        if change_stdout:
            self.stdout = sys.stdout
            sys.stdout = open('results/' + name + '_report.txt', encoding = 'utf-8', mode = 'w+')
            self.stderr = sys.stderr
            sys.stderr = open('results/' + name + '_error.txt', encoding = 'utf-8', mode = 'w+')

    def generate_report(self, name, clf, X_train, X_test, y_train, y_test, gridsearchcv = None):
        if gridsearchcv != None: print('Best params: ', gridsearchcv.best_params_)
        if gridsearchcv != None: print(gridsearchcv.cv_results_)
        print('Accuracy train: ', clf.score(X = X_train, y = y_train))
        print('Accuracy test: ', clf.score(X = X_test, y = y_test))

        y_train_pred = clf.predict(X = X_train)
        print(classification_report(y_train, y_train_pred, zero_division = 1))
        print(confusion_matrix(y_train, y_train_pred))

        y_test_pred = clf.predict(X = X_test)
        print(classification_report(y_test, y_test_pred, zero_division = 1))
        print(confusion_matrix(y_test, y_test_pred))

        if gridsearchcv != None:
            if self.iskeras(gridsearchcv.best_estimator_):
                print(gridsearchcv.best_estimator_.model.summary())

        print('='.join(['' for n in range(80)]))

    def close(self, change_stdout = True):
        if change_stdout:
            sys.stdout.close()
            sys.stdout = self.stdout
            sys.stderr.close()
            sys.stderr = self.stderr

    def gridsearchcv_graph(self, name, gridsearchcv):
        best_index_ = gridsearchcv.best_index_
        cv_results_ = gridsearchcv.cv_results_
        best_params_ = gridsearchcv.best_params_
        param_grid = gridsearchcv.param_grid
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
                    'images/' + name + '_' + param + '_fix',
                    dpi = 400
                )
                fig.clf()
                fig, ax = plt.subplots()
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
                    'images/' + name + '_' + param + '_no_fix',
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

    def load_model(self, name = 'dummy_model', sparse = True):
        models_availables = self.ls_model()
        to_load = [model for model in models_availables if name in model]
        assert len(to_load) > 0, 'No hay modelo para cargar'
        """file_in = open('models/model_' + name + '.pkl', mode = 'rb')
        clf = load(file_in)
        file_in.close()"""
        clf = joblib.load('models/model_' + name + '.pkl')
        if self.iskeras(clf['GridSearchCV'].best_estimator_):
            clf['GridSearchCV'].best_estimator_.model = tf.keras.models.load_model('models/model_' + name + '.h5')
            from tensorflow.keras import Input
            clf['GridSearchCV'].best_estimator_.model.layers[0] = Input(
                batch_size = 256,
                shape = 20000, 
                sparse = True
            )
            
        return (clf)

    def dump_model(self, name, clf, keras_model = None):
        key_save = ''
        model = None
        if keras_model != None:
            try: 
                os.remove('models/model_' + name + '.h5')
            except:
                pass
            keras_model.save('models/model_' + name + '.h5')
            if isinstance(clf,Pipeline):
                for key, value in clf.steps:
                    if isinstance(value,GridSearchCV):
                        if self.iskeras(value.best_estimator_):
                            model = clf[key].best_estimator_.model
                            clf[key].best_estimator_.model = None
                            key_save = key
        try:
            os.remove('models/model_' + name + '.pkl', mode = 'wb+')
        except:
            pass
        """file_out = open('models/model_' + name + '.pkl', mode = 'wb+')
        dump(clf, file_out)
        file_out.close()"""
        joblib.dump(clf, 'models/model_' + name + '.pkl')
        if keras_model != None:
            clf[key_save].best_estimator_.model = None
    
    def ls_model(self):
        models = []
        for _,_,files in os.walk('./models/'):
            for f in files:
                if '.pkl' in f or '.h5' in f:
                    models.append(f.replace('.pkl','').replace('.h5','').replace('model_',''))
        return(list(set(models)))