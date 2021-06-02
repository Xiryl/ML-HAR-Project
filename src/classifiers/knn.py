from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def knn(x_train, y_train, x_test, y_test, n_neighbors=1, p=1, metric='euclidean'):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    stats(y_test, y_pred)
    return


def knn_gs(x_train, y_train, x_test, y_test):
    tuned_parameters = [{'n_neighbors': [1, 3, 5],
                'p': [1, 3, 5],
                'metric': ['euclidean', 'manhattan']}]

    print("\t\t\t- Params: ", tuned_parameters)

    clf = GridSearchCV(
        KNeighborsClassifier(), tuned_parameters, scoring='accuracy'
    )
    clf.fit(x_train, y_train)

    print("\t\t\t- Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    model = KNeighborsClassifier()
    model.set_params(**clf.best_params_)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    stats(y_test, y_pred)
    return


def print_cmatrix(y_test, y_pred):
    cmatrix = confusion_matrix(y_test, y_pred, normalize='true')
    cm = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
    cm.plot()
    plt.show()


def stats(y_test, y_pred):
    prf1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("===== KNN ======")
    print("-Precision: ", prf1[0].round(2), "\n-Recall:    ", prf1[1].round(2), "\n-F1:        ", prf1[2].round(2), "\n- Accuracy:", accuracy)
    print("================")
    return prf1
