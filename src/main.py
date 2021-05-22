from utils import FileUtils, Preprocessing
from classifiers import svm

def run():
    df_train, df_test = FileUtils.load_dataset('./data/train.csv', './data/test.csv')

    # split data
    y_train = df_train.Activity
    x_train = df_train.drop(['subject', 'Activity'], axis=1)
    y_test = df_test.Activity
    x_test = df_test.drop(['subject', 'Activity'], axis=1)

    # encode labels
    y_train_encoded = Preprocessing.encode_labels(y_train)
    y_test_encoded = Preprocessing.encode_labels(y_test)

    # scale data
    x_train_s, x_test_s = Preprocessing.scale_dataset(x_train, x_test, scaler_type='standardscaler')

    svm.svm(x_train_s, y_train_encoded, x_test_s, y_test_encoded, max_iter=100, kernel='linear')


if __name__ == '__main__':
    run()

