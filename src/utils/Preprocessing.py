from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def encode_labels(y_train):
    """
    @brief Encode labels from string to int
    :param y_train: train labels
    :return: encoded labels
    """
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_train)
    y = encoder.transform(y_train)
    return y


def scale_dataset(x_train, x_test, scaler_type):
    """
    @brif scale data
    :param x_train: train
    :param x_test: test
    :param scaler_type: x_train_s, x_test_s scaled data
    :return:
    """
    scaler = StandardScaler()

    if scaler_type == 'minmaxscaler':
        scaler = MinMaxScaler()

    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.fit_transform(x_test)

    return x_train_s, x_test_s
