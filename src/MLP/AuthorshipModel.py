from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

from MLP.AuthorshipPreprocessing import AuthorshipPreprocessing
from MLP.mlp_model import mlp_model

class AuthorshipModel(BaseEstimator, ClassifierMixin):
    def __init__(self, labels ,k = 20, ngram_range = (1,2), max_features = 50, layers = 2, 
        units = 16, dropout_rate = 0.2, epochs = 300, verbose = False):
        self.verbose = verbose
        self.k = k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.preprocessing = AuthorshipPreprocessing(
            verbose = self.verbose,
            k = self.k,
            ngram_range = self.ngram_range,
            max_features = self.max_features
        )
        self.num_classes = len(labels)
        self.le = LabelEncoder()
        self.le.fit(labels)

    def fit(self, X, y):
        y = self.le.transform(y)
        X_pre = self.preprocessing.fit_transform(X,y)

        self.param_grid = {
            'layers': [2,3,4,5],
            'units': [16,32,64,128],
            'dropout_rate': [0.2,0.3,0.4,0.5]
        }

        self.clf = Pipeline([
            ('preprocessing', self.preprocessing),
            ('GridSearchCV', GridSearchCV(
                estimator = KerasClassifier(
                    mlp_model,
                    input_shape = (X_pre.shape[1],),
                    num_classes = self.num_classes,
                    epochs = self.epochs,
                    verbose = False
                ),
                cv = 5,
                n_jobs = 1,
                param_grid = self.param_grid,
                verbose = self.verbose
            ))
        ], verbose = self.verbose)

        self.clf.fit(X,y)


    def predict(self, X):
        X = self.preprocessing.transform(X)

    def score(self, X, y):
        """y = self.le.transform(y)
        X = self.preprocessing.transform(X)
        X.sort_indices()
        return(self.model.evaluate(X, y, verbose = False)[1])"""
        y = self.le.transform(y)
        return(self.clf.score(X,y))

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
    model = AuthorshipModel(
        labels = list(np.unique(y))
    )
    model.fit(X,y)