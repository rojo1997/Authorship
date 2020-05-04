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

from functions import real_xml, filter_df

from MLP.AuthorshipModel import AuthorshipModel

def main():
    random_state = 1
    nwords = 50
    frecuency = 200
    subset = None
    nfiles = None

    df = real_xml('./iniciativas08/', nfiles = nfiles)
    print(df.shape)
    if subset != None: df = df.sample(10000, random_state = random_state)
    print("Leido XML: ", df.shape)
    print(time.strftime("%X"))

    if nfiles == None: df = filter_df(df, nwords = nwords, frecuency = frecuency)
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

    clf = AuthorshipModel(
        labels = list(np.unique(df['name'])),
        k = 20000,
        max_features = 100000,
        units = 64,
        layers = 3,
        dropout_rate = 0.4,
        epochs = 50,
        verbose = True
    )

    print(time.strftime("%X"))
    clf.fit(X = X_train, y = y_train)
    print(time.strftime("%X"))

    print("best_score_", clf.clf['GridSearchCV'].best_score_)
    print("best_params_", clf.clf['GridSearchCV'].best_params_)

    print("Accuracy train: ", clf.score(X = X_train, y = y_train))
    print("Accuracy test: ", clf.score(X = X_test, y = y_test))
    
    print(time.strftime("%X"))

    """y_test_pred = clf.predict(X = X_test)
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
        pickle.dump(clf, file)"""

if __name__ == "__main__":
    main()