from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tsfel as ts
import warnings
from scipy import stats
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def encode_labels(df):
    labels = df["activity"]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    df["activity"] = encoder.transform(labels)
    return df


def scale_dataset(_config, x_train, x_test):
    norm_type = _config['NORMALIZATION']['norm_type']
    scaler = StandardScaler()

    if norm_type == 'minmax':
        scaler = MinMaxScaler()

    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.fit_transform(x_test)

    return x_train_s, x_test_s


def data_treatment(_config, df):
    treatment_type = _config['DATA_REPRESENTATION']['treatment_type']

    if treatment_type == "feat_extraction":
        df = apply_feat_extraction(_config, df)

    return df


def apply_feat_extraction(_config, df):
    sampling_frequency = int(_config['DATA_REPRESENTATION']['sampling_frequency'])
    time_window_len = int(_config['DATA_REPRESENTATION']['time_window_len'])
    window_overlap = int(_config['DATA_REPRESENTATION']['window_overlap'])
    features_domain = _config['DATA_REPRESENTATION']['features_domain']

    x = df.drop(['activity', 'user', 'timestamp'], axis=1)
    y = df['activity']

    if features_domain == "all":
        cfg = ts.get_features_by_domain()
    else:
        cfg = ts.get_features_by_domain(features_domain)

    tsfel_overlap = round(window_overlap / time_window_len, 3)

    time_window_size = int(sampling_frequency * time_window_len)

    X_features = ts.time_series_features_extractor(cfg,
                                                   x,
                                                   fs=sampling_frequency,
                                                   window_size=time_window_size,
                                                   overlap=tsfel_overlap,
                                                   verbose=1)

    Y_features = labels_windowing(y, sampling_frequency, time_window_size, window_overlap)

    df_features = pd.DataFrame(X_features)
    df_features['activity'] = Y_features

    return df_features


def labels_windowing(labels, sampling_frequency, time_window, overlap):
    hop_size = time_window - int(sampling_frequency * overlap)
    windowed_labels = list()

    for i in range(0, len(labels) - time_window + 1, hop_size):
        new_label = stats.mode(labels.values[i: i + time_window])[0][0]
        windowed_labels.append(new_label)

    return np.asarray(windowed_labels)


def do_train_test_split(_config, df):
    test_size = float(_config['TRAINING']['test_size'])
    y = df["activity"]
    x = df.drop("activity", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=True,
        test_size=test_size,
        random_state=28)

    print('\t\t- test_size: ', test_size)
    print('\t\t- x_train: ', x_train.shape)
    print('\t\t- y_train: ', y_train.shape)
    print('\t\t- x_test: ', x_test.shape)
    print('\t\t- y_test: ', y_test.shape)

    return x_train, x_test, y_train, y_test