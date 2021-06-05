import warnings

import numpy as np
import pandas as pd
import tsfel as ts
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore")


def encode_labels(df):
    """
    Label encoding
    """
    labels = df["activity"]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    integer_mapping = {l: i for i, l in enumerate(encoder.classes_)}
    print("\t\t- Mapping:", integer_mapping)
    df["activity"] = encoder.transform(labels)
    return df


def scale_dataset(_config, x_train, x_test):
    """
    Scale dataset values
    """
    norm_type = _config['NORMALIZATION']['norm_type']
    scaler = StandardScaler()
    print("\t\t- Apply: ", norm_type)

    if norm_type == 'minmax':
        scaler = MinMaxScaler()

    if norm_type == 'robust':
        scaler = RobustScaler()

    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.fit_transform(x_test)

    return x_train_s, x_test_s


def data_treatment(_config, df):
    """
    Apply data treatment (feature extraction)
    """
    treatment_type = _config['DATA_REPRESENTATION']['treatment_type']

    if treatment_type == "feat_extraction":
        df = apply_feat_extraction(_config, df)

    return df


def apply_feat_extraction(_config, df):
    """
    Apply feature extraction to the dataset
    """

    sampling_frequency = int(_config['DATA_REPRESENTATION']['sampling_frequency'])
    time_window_len = int(_config['DATA_REPRESENTATION']['time_window_len'])
    window_overlap = int(_config['DATA_REPRESENTATION']['window_overlap'])
    features_domain = _config['DATA_REPRESENTATION']['features_domain']

    x = df.drop(['activity', 'user', 'timestamp'], axis=1)
    # x = df.drop(['activity', 'timestamp'], axis=1)
    y = df['activity']

    if features_domain == "all":
        print("\t\t- Domain: 'all'")
        cfg = ts.get_features_by_domain()
    else:
        print("\t\t- Domain: '", features_domain, "'")
        cfg = ts.get_features_by_domain(features_domain)

    tsfel_overlap = round(window_overlap / time_window_len, 3)

    time_window_size = int(sampling_frequency * time_window_len)

    X_features = ts.time_series_features_extractor(cfg,
                                                   x,
                                                   fs=sampling_frequency,
                                                   window_size=time_window_size,
                                                   overlap=tsfel_overlap,
                                                   verbose=1)

    # Handling eventual missing values from the feature extraction
    X_features = fill_missing_values_after_feat_extraction(X_features)

    Y_features = labels_windowing(y, sampling_frequency, time_window_size, window_overlap)

    df_features = pd.DataFrame(X_features)
    df_features['activity'] = Y_features

    print("\t\t- New shape: ", df_features.shape)

    return df_features


def fill_missing_values_after_feat_extraction(df):
    """
    Fix missing values
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def labels_windowing(labels, sampling_frequency, time_window, overlap):
    """
    Apply sliding window to the labels
    """

    hop_size = time_window - int(sampling_frequency * overlap)
    windowed_labels = list()

    for i in range(0, len(labels) - time_window + 1, hop_size):
        new_label = stats.mode(labels.values[i: i + time_window])[0][0]
        windowed_labels.append(new_label)

    return np.asarray(windowed_labels)


def do_train_test_split(_config, df):
    """
    Split train and test dataset
    """

    test_size = float(_config['TRAINING']['test_size'])
    y = df["activity"]
    x = df.drop("activity", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=False,
        test_size=test_size,
        random_state=28)

    print('\t\t- test_size: ', test_size)
    print('\t\t- x_train: ', x_train.shape)
    print('\t\t- y_train: ', y_train.shape)
    print('\t\t- x_test: ', x_test.shape)
    print('\t\t- y_test: ', y_test.shape)

    return x_train, x_test, y_train, y_test


def do_features_selection(_config, x_train, x_test):
    """
    Do feature selection
    """

    feat_sel_type = _config['FEATURES_SELECTION']['feat_sel_type']
    print("\t\t- Apply: ", feat_sel_type)

    if feat_sel_type == 'variance':
        selector = VarianceThreshold()
        x_train_s = selector.fit_transform(x_train)
        x_test_s = selector.transform(x_test)

        print("\t\t- New shape x_train: ", x_train_s.shape)
        print("\t\t- New shape x_test : ", x_test_s.shape)
        return x_train_s, x_test_s


def do_balancing(_config, x_train, y_train):
    """
    do dataset balancing
    """

    balancing_technique = _config['BALANCING']['balancing_technique']
    print("\t\t- Apply: ", balancing_technique)

    if balancing_technique == 'under':
        sampler = RandomUnderSampler()
        x_train_b, y_train_b = sampler.fit_resample(x_train, y_train)

        print("\t\t- New shape x_train: ", x_train_b.shape)
        print("\t\t- New shape y_train : ", y_train_b.shape)
        return x_train_b, y_train_b
    else:
        return x_train, y_train


def data_filtering(_config, df_data):
    # Extract activity and user from original dataset
    df_activity_user = df_data[['activity', 'user']]
    # Get uniques couple of activity-user
    df_activity_user_uniques = df_activity_user.drop_duplicates()

    # use indexes a new column
    df_idx_activities_per_user = df_activity_user_uniques.reset_index()
    df_idx_activities_per_user.columns = ['idx', 'activity', 'user']

    ###
    # What next section do?
    # Now I cycle over the new idx, which contains the start-stop index of each activity for each user
    # I extract the subdataset of that specific activity of the users, and I apply the fft filter on it
    # Lather, I concat the small dataset filtered in one.
    ###

    filtered_dataset = []

    # cycle over the unique activities
    for i in range(0, df_idx_activities_per_user.shape[0]):
        # check for outOfBoundException
        if i != df_idx_activities_per_user.shape[0] - 1:
            # Take the sub dataframe to manage starting from idx (i) to last available index of that activity (i+1)
            tmp_data = df_data[df_idx_activities_per_user.idx.iloc[i]:df_idx_activities_per_user.idx.iloc[i + 1]]
        else:
            # Take the sub dataframe to manage starting from idx to last available row
            tmp_data = df_data[df_idx_activities_per_user.idx.iloc[i]:]

    #
    # filter_dataset = []
    # feat_dataset = []
    # for i in range(0, ddd.shape[0]):
    #     if i != ddd.shape[0] - 1:
    #         tmp_data = data[ddd.inde.iloc[i]:ddd.inde.iloc[i + 1]]  # devo andare fino al ultima riga del mio dataset
    #     else:
    #         tmp_data = data[ddd.inde.iloc[i]:]  # devo andare fino al ultima riga del mio dataset
    #     tmp_data.drop(['Time', 'A', 'P'], axis=1, inplace=True)
    #     tmp_data = filter(tmp_data)
    #     features_data = tsf(tmp_data, Wind, overla)
    #     features_data['A'] = ddd.A.iloc[i]
    #     features_data['P'] = ddd.P.iloc[i]
    #
    #     tmp_data['A'] = ddd.A.iloc[i]
    #     tmp_data['P'] = ddd.P.iloc[i]
    #     filter_dataset.append(tmp_data)
    #     feat_dataset.append(features_data)

    return df_data


def feat_extraction(_config, df_data):
    """
    Apply feature extraction to the dataset
    """
    # Extract activity and user from original dataset
    df_activity_user = df_data[['activity', 'user']]
    # Get uniques couple of activity-user
    df_activity_user_uniques = df_activity_user.drop_duplicates()

    # use indexes a new column
    df_idx_activities_per_user = df_activity_user_uniques.reset_index()
    df_idx_activities_per_user.columns = ['idx', 'activity', 'user']

    df_data = df_data.drop(['activity', 'user', 'timestamp'], axis=1)

    sampling_frequency = int(_config['DATA_REPRESENTATION']['sampling_frequency'])
    time_window_len = int(_config['DATA_REPRESENTATION']['time_window_len'])
    window_overlap = int(_config['DATA_REPRESENTATION']['window_overlap'])
    features_domain = _config['DATA_REPRESENTATION']['features_domain']

    if features_domain == "all":
        print("\t\t- Domain: 'all'")
        cfg = ts.get_features_by_domain()
    else:
        print("\t\t- Domain: '", features_domain, "'")
        cfg = ts.get_features_by_domain(features_domain)

    tsfel_overlap = round(window_overlap / time_window_len, 3)

    time_window_size = int(sampling_frequency * time_window_len)

    feat_dataset = []
    qtaSkipped = 0
    # cycle over the unique activities
    for i in range(0, df_idx_activities_per_user.shape[0]):
        print( i, "/", df_idx_activities_per_user.shape[0])

        # check for outOfBoundException
        if i != df_idx_activities_per_user.shape[0] - 1:
            # Take the sub dataframe to manage starting from idx (i) to last available index of that activity (i+1)
            tmp_data = df_data[df_idx_activities_per_user.idx.iloc[i]:df_idx_activities_per_user.idx.iloc[i + 1]]
        else:
            # Take the sub dataframe to manage starting from idx to last available row
            tmp_data = df_data[df_idx_activities_per_user.idx.iloc[i]:]

        if tmp_data.shape[0] < time_window_size:
            qtaSkipped = qtaSkipped + 1
            print("Skipping [", i, "] -> qtaSkipped:", qtaSkipped)
            continue

        X_features = ts.time_series_features_extractor(cfg,
                                                   tmp_data,
                                                   fs=sampling_frequency,
                                                   window_size=time_window_size,
                                                   overlap=tsfel_overlap,
                                                   verbose=0)

        # Handling eventual missing values from the feature extraction
        X_features = fill_missing_values_after_feat_extraction(X_features)

        X_features['activity'] = df_idx_activities_per_user.activity.iloc[i]
        X_features['user'] = df_idx_activities_per_user.user.iloc[i]

        feat_dataset.append(X_features)


    df_feat = pd.concat(feat_dataset, axis=0)
    print("\t\t- Old shape: ", df_data.shape)
    print("\t\t- New shape: ", df_feat.shape)

    return df_feat
