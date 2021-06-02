from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV

def svm(x_train, y_train, x_test, y_test, max_iter=1000, kernel='linear'):
    model = SVC(kernel=kernel)
    model = train(model, x_train, y_train)
    predicted = predict(model, x_test)
    print_cmatrix(predicted, y_test)
    stats(predicted, y_test)
    return predicted


def svm_gs(x_train, y_train, x_test, y_test):
    # tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'],
    #                      'gamma': ['auto', 'scale'],
    #                      'C': [1, 10, 100, 1000],
    #                      'max_iter': [300, 500, 1000]}]

    # tuned_parameters = [{'C': [0.01, 0.1, 1, 10],
    #             'kernel': ['linear', 'poly', 'rbf'],
    #             'gamma': ['scale', 'auto']}]

    tuned_parameters = [{'C': [0.01],
                'kernel': ['linear']}]


    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='accuracy'
    )
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    model = SVC()
    model.set_params(**clf.best_params_)
    model = train(model, x_train, y_train)
    predicted = predict(model, x_test)
    print_cmatrix(predicted, y_test)
    stats(predicted, y_test)
    return predicted

def train(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    predicted = model.predict(x_test)
    return predicted


def print_cmatrix(predicted, y_test):
    cmatrix = confusion_matrix(y_test, predicted)
    print(cmatrix)
    sns.heatmap(cmatrix / np.sum(cmatrix), annot=True,
                fmt='.2%', cmap='Blues')
    plt.show()
    return cmatrix


def stats(y_pred, y_true):
    prf1 = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print("===== SVM ======")
    print("-Precision: ", prf1[0].round(2), "\n-Recall:    ", prf1[1].round(2), "\n-F1:        ", prf1[2].round(2))
    print("================")
    return prf1
