from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NVarPrint(BaseEstimator, TransformerMixin):
    def __init__(self, verbose = False):
        self.verbose = verbose
        pass

    def fit(self, X, y=None):
        if self.verbose:
            print("sparcity: ", 1.0 - len(X.indices) / np.prod(X.shape))
            print("density: ", len(X.indices) / np.prod(X.shape))
        return(self)

    def transform(self,X):
        if self.verbose: print("NVarPrint: ", X.shape)
        return(X)