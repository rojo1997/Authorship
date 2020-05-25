from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import SelectKBest, f_classif

class ANOVA(BaseEstimator, TransformerMixin):
    def __init__(self, k = 1000):
        self.k = k
        self.ANOVA = SelectKBest(
            score_func = f_classif, 
            k = self.k
        )

    def fit(self, X, y):
        if X.shape[1] < self.k:
            self.ANOVA = SelectKBest(
                score_func = f_classif, 
                k = X.shape[1]
            )
        self.ANOVA.fit(X,y)
        return(self)

    def transform(self, X):
        X = self.ANOVA.transform(X)
        X.sort_indices()
        return(X)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))

class Kendall(BaseEstimator, TransformerMixin):
    def __init__(self, k = 1000):
        self.k = k
        self.features = []

    def fit(self, X, y):
        corr_matriz = X.corr(method = 'kendall')
        return(self)

    def transform(self, X):
        return(X)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return(self.transform(X))