import configparser

from classifiers import MLClassifiers
from utils import FileUtils, Preprocessing, PrintUtils, DataCleaning
import pandas as pd

_config = configparser.ConfigParser()
_config.read('config.ini')
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


def run():

    # --- Load dataset  ---
    print("\t - Load Dataset ...")
    df_data = FileUtils.load_dataset(_config)
    # -----------------

    if _config['DATASET']['dataset'] != 'local_preprocessed':
        # --- Print init stats  ---
        print("\t - Generating stats ...")
        if _config['INIT']['verbose'] == "True":
            PrintUtils.print_init_stats(_config['DATASET']['dataset'], df_data)

        generate_stats = _config['INIT']['generate_stats']
        if generate_stats == "True":
            PrintUtils.plot_count_per_subject(df_data)
            PrintUtils.plot_samplings_per_class(df_data)
            PrintUtils.plot_sampling_per_class_per_user(df_data)
            PrintUtils.plot_activity("Sitting", df_data)
            PrintUtils.plot_activity("Walking", df_data)
            PrintUtils.plot_activity("Jogging", df_data)
        # -----------------

        # --- Data Cleaning  ---
        print("\t - Data cleaning (fix NaN values) ...")
        df_data = DataCleaning.handle_nan_values(_config, df_data)
        # -----------------

        # --- Encode labels  ---
        print("\t - Encoding labels ...")
        df_data = Preprocessing.encode_labels(df_data)
        # -----------------

        # # --- Data Filtering v2 ---
        # print("\t - Data filtering (fft) v2 ...")
        # df_data = Preprocessing.data_filtering(_config, df_data)
        # # -----------------

        # # --- Data Filtering  ---
        # print("\t - Data cleaning (noise removal) ...")
        # df_data = DataCleaning.apply_filter(_config, df_data)
        # # -----------------

        # --- Data Treatment  ---
        print("\t - Data Treatment (feat extraction) ...")
        df_data = Preprocessing.feat_extraction(_config, df_data)

        print("\t - Save df_feature ...")
        FileUtils.save_dataset(df_data)
        # -----------------
    else:
        # --- Print init stats  ---
        print("\t - Generating stats ...")
        if _config['INIT']['verbose'] == "True":
            PrintUtils.print_init_stats(_config['DATASET']['dataset'], df_data)
        # -----------------

    PrintUtils.print_init_stats(_config['DATASET']['dataset'], df_data)
    # --- Train/Test Split  ---
    print("\t - Train/Test Split ...")
    # x_train, x_test, y_train, y_test = Preprocessing.do_train_test_split(_config, df_data)

    df_test_1 = df_data.loc[df_data['user'] == 8]
    df_test_2 = df_data.loc[df_data['user'] == 20]
    frames = [df_test_1, df_test_2]
    df_test = pd.concat(frames, axis=0)
    df_train = df_data.loc[df_data['user'] != 8]
    df_train = df_train.loc[df_train['user'] != 20]

    df_train = df_train.drop('user', axis=1)
    y_train = df_train['activity']
    df_train = df_train.drop('activity', axis=1)
    x_train = df_train

    df_test = df_test.drop('user', axis=1)
    y_test = df_test['activity']
    df_test = df_test.drop('activity', axis=1)
    x_test = df_test
    # -----------------

    # --- Normalization  ---
    print("\t - Normalization ...")
    x_train, x_test = Preprocessing.scale_dataset(_config, x_train, x_test)
    # -----------------

    tsne = TSNE(random_state=42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(df_train)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_train, palette="bright")
    plt.show()

    # --- Features Selection  ---
    print("\t - Features Selection ...")
    x_train, x_test = Preprocessing.do_features_selection(_config, x_train, x_test)
    # -----------------

    # # --- Data Balancing  ---
    # print("\t - Data Balancing ...")
    # x_train, y_train = Preprocessing.do_balancing(_config, x_train, y_train)
    # # -----------------

    # --- Model execution: ML  ---
    print("\t - Model execution: ML ...")
    MLClassifiers.run_classifiers(_config, x_train, y_train, x_test, y_test)
    # -----------------


if __name__ == '__main__':
    run()
