from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from utils import Preprocessing


def knn(x_train, y_train, x_test, y_test, n_neighbors=1, p=1, metric='euclidean'):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_cmatrix(y_test, y_pred)
    f1 = stats(y_test, y_pred)
    return f1[2].round(2)


def knn_gs(x_train, y_train, x_test, y_test):
    tuned_parameters = [{'n_neighbors': list(range(1, 11)),
                        'p': [1, 3, 5],
                        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}]

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
    cm = ConfusionMatrixDisplay(confusion_matrix=cmatrix,
                                display_labels=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    cm.plot(colorbar=True, cmap='Blues')
    plt.title("KNN")
    plt.show()


def stats(y_test, y_pred):
    prf1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("\t\t\t===== KNN ======")
    print("\t\t\t-Precision: ", prf1[0].round(2), "\n\t\t\t-Recall:    ", prf1[1].round(2), "\n\t\t\t-F1:        ",
          prf1[2].round(2), "\n\t\t\t-Accuracy:  ", accuracy.round(2))
    print("\t\t\t================")
    return prf1


def knn_kfold(_config, df_data, n_neighbors=5, metric='euclidean', kfold=3):
    df_data_y = df_data['activity']
    df_data = df_data.drop('activity', axis=1)
    df_data = df_data.drop('user', axis=1)
    df_data_x = df_data

    kf = KFold(n_splits=kfold)

    for train_index, test_index in kf.split(df_data_x):
        x_train, x_test = df_data_x.iloc[train_index], df_data_x.iloc[test_index]
        y_train, y_test = df_data_y.iloc[train_index], df_data_y.iloc[test_index]

        # --- Normalization  ---
        print("\t - Normalization ...")
        x_train, x_test = Preprocessing.scale_dataset(_config, x_train, x_test)
        # -----------------

        # --- Features Selection  ---
        print("\t - Features Selection ...")
        x_train, x_test = Preprocessing.do_features_selection(_config, x_train, x_test)
        # -----------------

        # --- Data Balancing  ---
        print("\t - Data Balancing ...")
        x_train, y_train = Preprocessing.do_balancing(_config, x_train, y_train)
        # -----------------

        f1_score = knn(x_train, y_train, x_test, y_test, n_neighbors=n_neighbors, p=1, metric=metric)
        print("--->>> F1 :", f1_score)

    return None
