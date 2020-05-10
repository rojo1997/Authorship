from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

from Preprocessing.TFIDFVectorizerANOVA import TFIDFVectorizerANOVA
from TFIDFANOVAMLP.mlp_model import mlp_model

class Authorship(BaseEstimator, ClassifierMixin):
    def __init__(self, labels ,k = 20000, ngram_range = (1,2), max_features = 100000, layers = 2, 
        units = 96, dropout_rate = 0.3, epochs = 50, verbose = False):
        self.verbose = verbose
        self.k = k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.num_classes = len(labels)
        self.preprocessing = TFIDFVectorizerANOVA(
            verbose = self.verbose,
            k = self.k,
            ngram_range = self.ngram_range,
            max_features = self.max_features
        )
        self.param_grid = {
            'units': [self.units - 32,self.units, self.units + 32],
            'dropout_rate': [0.2,0.3,0.4],
        }
        """self.clf = Pipeline([
            ('preprocessing', self.preprocessing),
            ('GridSearchCV', GridSearchCV(
                estimator = KerasClassifier(
                    mlp_model,
                    input_shape = self.k,
                    num_classes = self.num_classes,
                    layers = self.layers,
                    units = self.units,
                    dropout_rate = self.dropout_rate,
                    epochs = self.epochs,
                    verbose = False
                ),
                cv = 5,
                n_jobs = 1,
                param_grid = self.param_grid,
                verbose = self.verbose
            ))
        ], verbose = self.verbose)"""
        self.clf = Pipeline([
            ('preprocessing', self.preprocessing),
            ('mlp_model', KerasClassifier(
                mlp_model,
                input_shape = self.k,
                num_classes = self.num_classes,
                layers = self.layers,
                units = self.units,
                dropout_rate = self.dropout_rate,
                epochs = self.epochs,
                batch_size = 1024,
                verbose = False
            ))
        ], verbose = self.verbose)
        
        
        self.le = LabelEncoder()
        self.le.fit(labels)

    def fit(self, X, y):
        self.clf.fit(X,y = self.le.transform(y))


    def predict(self, X):
        return(self.le.inverse_transform(self.clf.predict(X)))

    def score(self, X, y):
        return(self.clf.score(X,y = self.le.transform(y)))

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
    model = Authorship(
        labels = list(np.unique(y))
    )
    model.fit(X,y)