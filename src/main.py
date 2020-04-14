###############################################################################


###############################################################################

import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
import dill as pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from Authorship import Authorship
from functions import readData

def filter(df, nphrases = 5, frecuency = 10, inplace = False):
    if inplace:
        df.drop(df[df.nphrases >= nphrases].index, inplace = True)
        groups = df.groupby(['name']).count()
        df["gr"] = df.name.map(lambda n: float(groups.loc[n,"text"]))
        df = df.drop(df[df.gr >= frecuency].index, inplace = True)
    else:
        df_copy = df.copy()
        df_copy = df_copy[df_copy.nphrases >= nphrases]
        groups = df_copy.groupby(['name']).count()
        df_copy["gr"] = df_copy.name.map(lambda n: float(groups.loc[n,"text"]))
        df_copy = df_copy[df_copy.gr >= frecuency]
        return(df_copy)

def main():
    random_state = 1
    nphrases = 5
    frecuency = 10

    """data = readData('./iniciativas08/')
    print("Leido XML: ", data.shape)
    print(time.strftime("%X"))"""
    
    print(time.strftime("%X"))
    df = pd.read_csv(
        "data/data.csv", 
        sep='\t', 
        encoding='utf-8',
        #nrows = 10000
    )
    print(time.strftime("%X"))
    print("Leido CSV: ", df.shape)
    #print("Columns: ", df.columns)

    """data.to_csv("data.csv", sep='\t', encoding='utf-8', index = None)
    print("Guardado CSV")
    print(time.strftime("%X"))"""

    df = filter(df, nphrases = nphrases,frecuency = frecuency, inplace = False)
    print("Filtro sobre Dataset: ", df.shape)

    le = LabelEncoder()
    le.fit(df['name'])
    #print("Etiquetas codificadas: ", le.classes_)

    print('Numero de documentos: ', df.shape[0])
    print('Etiquetas: ', len(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['name'],
        test_size = 0.2,
        random_state = random_state
    )
    print("Train: ", X_train.shape)
    print("Test: ", X_test.shape)

    clf = Authorship(
        verbose = 1,
        random_state = random_state,
        le = le
    )

    print(time.strftime("%X"))
    clf.fit(X = X_train, y = y_train)
    print(time.strftime("%X"))

    print("Accuracy train: ", clf.score(X = X_train.text, y = y_train))
    print("Accuracy test: ", clf.score(X = X_test.text, y = y_test))
    
    print(time.strftime("%X"))

    predicted = clf.predict(X = X_test.text)
    print(classification_report(y_test, predicted))
    print(confusion_matrix(y_test, predicted))

    with open('model/classifier.pkl', 'wb') as file:
        pickle.dump(clf, file)

if __name__ == "__main__":
    main()