from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def svm(x_train, y_train, x_test, y_test, max_iter=100, kernel='linear'):
    model = SVC(kernel=kernel, max_iter=max_iter)
    model = train(model, x_train, y_train)
    predicted = predict(model, x_test)
    print_cmatrix(predicted, y_test)
    return predicted


def train(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    predicted = model.predict(x_test)
    return predicted


def print_cmatrix(predicted, y_test):
    conf_matrix = confusion_matrix(y_test, predicted)
    print(conf_matrix)
    plt.imshow(conf_matrix)
    plt.show()
    return