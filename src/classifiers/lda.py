from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from utils import Preprocessing


def lda(x_train, y_train, x_test, y_test):
    model = LinearDiscriminantAnalysis(solver="svd", store_covariance=True, tol=0.0001)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    stats(y_test, y_pred)
    return


def lda_gs(x_train, y_train, x_test, y_test):

    tuned_parameters = [{'solver': ['svd'],
                        'store_covariance': ['True', 'False'],
                        'tol': [ 0.0001, 0.001, 0.01, 0.1]}]

    print("\t\t\t- Params: ", tuned_parameters)

    clf = GridSearchCV(
        LinearDiscriminantAnalysis(), tuned_parameters, scoring='accuracy'
    )
    clf.fit(x_train, y_train)

    print("\t\t\t- Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    model = LinearDiscriminantAnalysis()
    model.set_params(**clf.best_params_)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    stats(y_test, y_pred)
    return


def print_cmatrix(y_test, y_pred):
    cmatrix = confusion_matrix(y_test, y_pred, normalize='true')
    cm = ConfusionMatrixDisplay(confusion_matrix=cmatrix,
                                display_labels=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    cm.plot(colorbar=True, cmap='Blues')
    plt.title("LDA")
    plt.show()


def stats(y_test, y_pred):
    prf1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("\t\t\t===== LDA ======")
    print("\t\t\t-Precision: ", prf1[0].round(2), "\n\t\t\t-Recall:    ", prf1[1].round(2), "\n\t\t\t-F1:        ",
          prf1[2].round(2), "\n\t\t\t-Accuracy:  ", accuracy.round(2))
    print("\t\t\t================")
    return prf1

