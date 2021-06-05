from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV


def svm(x_train, y_train, x_test, y_test, kernel='linear', gamma='auto', C=0.1, max_iter=1000):
    model = SVC(kernel=kernel, gamma=gamma, C=C, max_iter=max_iter)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    stats(y_test, y_pred)
    return


def svm_gs(x_train, y_train, x_test, y_test):
    tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'],
                         'gamma': ['auto', 'scale'],
                         'C': [1, 10, 100, 1000],
                         'max_iter': [300, 500, 1000]}]

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='accuracy'
    )

    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    model = SVC()
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
    plt.title("SVM")
    plt.show()


def stats(y_test, y_pred):
    prf1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("\t\t\t===== SVM ======")
    print("\t\t\t-Precision: ", prf1[0].round(2), "\n\t\t\t-Recall:    ", prf1[1].round(2), "\n\t\t\t-F1:        ", prf1[2].round(2), "\n\t\t\t-Accuracy:  ", accuracy.round(2))
    print("\t\t\t================")
    return prf1



