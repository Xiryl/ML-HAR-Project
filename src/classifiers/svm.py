from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def svm(x_train, y_train, x_test, y_test, max_iter=100, kernel='linear'):
    model = SVC(kernel=kernel, max_iter=max_iter)
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
    plt.imshow(cmatrix)
    plt.show()
    return cmatrix


def stats(y_pred, y_true):
    prf1 = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print("===== SVM ======")
    print("-Precision: ", prf1[0].round(2), "\n-Recall:    ", prf1[1].round(2), "\n-F1:        ", prf1[2].round(2))
    print("================")
    return prf1
