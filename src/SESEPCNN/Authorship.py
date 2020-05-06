from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

from Preprocessing.Sequences import Sequences
from SESEPCNN.sepcnn_model import sepcnn_model

class Authorship(BaseEstimator, ClassifierMixin):
    def __init__(self, labels, maxlen = 200, num_words = 20000, dropout_rate = 0.2, epochs = 50, verbose = False):
        self.verbose = verbose
        self.maxlen = maxlen
        self.num_words = num_words
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.num_classes = len(labels)
        self.preprocessing = Sequences(
            verbose = self.verbose,
            num_words = self.num_words,
            maxlen = self.maxlen
        )
        self.param_grid = {
            'dropout_rate': [0.1,0.2]
        }
        self.clf = Pipeline([
            ('preprocessing', self.preprocessing),
            ('GridSearchCV', GridSearchCV(
                estimator = KerasClassifier(
                    sepcnn_model,
                    dropout_rate = self.dropout_rate,
                    input_shape = (self.maxlen,),
                    num_classes = self.num_classes,
                    num_features = self.num_words,
                    epochs = self.epochs,
                    verbose = True
                ),
                cv = 5,
                n_jobs = 1,
                param_grid = self.param_grid,
                verbose = self.verbose
            ))
        ], verbose = self.verbose)
        
        
        self.le = LabelEncoder()
        self.le.fit(labels)

    def fit(self, X, y):
        self.clf.fit(X,y = self.le.transform(y))


    def predict(self, X):
        return(self.le.inverse_transform(
            np.argmax(self.clf.predict(X), axis = 0)
        ))

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