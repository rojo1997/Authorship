###############################################################################


###############################################################################

import time
import pandas as pd
import numpy as np
import string
import dill as pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from Authorship import Authorship
from functions import real_xml, filter_df

def main():
    random_state = 1
    nwords = 20
    frecuency = 50

    df = real_xml('./iniciativas08/').sample(10000, random_state = random_state)
    print("Leido XML: ", df.shape)
    print(time.strftime("%X"))
    
    """print(time.strftime("%X"))
    df = pd.read_csv(
        "data/data.csv", 
        sep='\t', 
        encoding='utf-8',
        #nrows = 800
    )
    print(time.strftime("%X"))
    print("Leido CSV: ", df.shape)"""
    #print("Columns: ", df.columns)

    """data.to_csv("data.csv", sep='\t', encoding='utf-8', index = None)
    print("Guardado CSV")
    print(time.strftime("%X"))"""

    df = filter_df(df, nwords = nwords, frecuency = frecuency)
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
        le = le,
        n_jobs = 3
    )

    print(time.strftime("%X"))
    clf.fit(X = X_train, y = y_train)
    print(time.strftime("%X"))

    print("Accuracy train: ", clf.score(X = X_train, y = y_train))
    print("Accuracy test: ", clf.score(X = X_test, y = y_test))
    
    print(time.strftime("%X"))

    y_test_pred = clf.predict(X = X_test)
    print(classification_report(y_test, y_test_pred))
    report = classification_report(y_test, y_test_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('data/report.txt')
    print(confusion_matrix(y_test, y_test_pred))
    np.savetxt('data/confusion_matrix_normalize.txt', confusion_matrix(y_test, y_test_pred), delimiter=',')
    print(confusion_matrix(y_test, y_test_pred))
    np.savetxt('data/confusion_matrix.txt', confusion_matrix(y_test, y_test_pred), delimiter=',')

    print("best_score_", clf.clf.best_score_)
    print("best_params_", clf.clf.best_params_)
    print("cv_results_", clf.clf.cv_results_)
    print("cv", clf.clf.cv)

    with open('model/classifier.pkl', 'wb+') as file:
        pickle.dump(clf, file)

if __name__ == "__main__":
    main()