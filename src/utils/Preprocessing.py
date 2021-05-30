from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


def encode_labels(df):
    labels = df["activity"]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    df["activity"] = encoder.transform(labels)
    return df


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
