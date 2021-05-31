from sklearn.feature_selection import VarianceThreshold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

from utils import FileUtils, Preprocessing, PrintUtils, DataCleaning
from classifiers import svm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import configparser

_config = configparser.ConfigParser()
_config.read('config.ini')


def print_config(_config):
    generate_stats = _config['INIT']['generate_stats']
    filter = _config['DATA_CLEANING']['filter']
    representation_type = _config['DATA_REPRESENTATION']['treatment_type']

    if generate_stats == "True":
        print("Executing using [ generate stats ] as init phase.")
    else:
        print("Executing using [ no generate stats ] as init phase.")

    if filter == "none":
        print("Executing using [ no FFT filter ] as data cleaning type.")
    else:
        print("Executing using [ FFT filter: low ] as data cleaning type.")

    print("Executing using [", representation_type, "] as data treatment type.")
    return


def run():

    # --- Print running configuration  ---
    print_config(_config)
    # -----------------

    print("-----------------")

    # --- Load datasets  ---
    print("\t - Load Dataset ...")
    df_data = FileUtils.load_dataset(_config)
    # -----------------

    if _config['DATASET']['dataset'] != 'local_preprocessed':
        # --- Print init stats  ---
        print("\t - Generating stats ...")
        generate_stats = _config['INIT']['generate_stats']
        if generate_stats == "True":
            PrintUtils.print_init_stats("WISDM_v1", df_data)
            PrintUtils.plot_count_per_subject(df_data)
            PrintUtils.plot_samplings_per_class(df_data)
            PrintUtils.plot_sampling_per_class_per_user(df_data)
            PrintUtils.plot_activity("Sitting", df_data)
            PrintUtils.plot_activity("Walking", df_data)
            PrintUtils.plot_activity("Jogging", df_data)
        # -----------------

        # --- Data FFT filtering  ---
        print("\t - Data filtering (noise removal) ...")
        df_data_filtered = DataCleaning.apply_filter(_config, df_data)
        df_data = df_data_filtered
        # -----------------

        # --- Encode labels  ---
        print("\t - Encoding labels ...")
        df_data_encoded = Preprocessing.encode_labels(df_data)
        df_data = df_data_encoded
        # -----------------

        # --- Data Treatment  ---
        print("\t - Data Treatment ...")
        df_data = Preprocessing.data_treatment(_config, df_data)

        print("\t - Save df_feature ...")
        FileUtils.save_dataset(df_data)
        # -----------------

    # --- Train/Test Split  ---
    print("\t - Train/Test Split ...")
    x_train, x_test, y_train, y_test = Preprocessing.do_train_test_split(_config, df_data)
    # -----------------

    # --- Normalization  ---
    print("\t - Normalization ...")
    x_train, x_test = Preprocessing.scale_dataset(_config, x_train, x_test)
    # -----------------

    # --- Features Selection  ---
    print("\t - Features Selection ...")
    x_train, x_test = Preprocessing.do_features_selection(_config, x_train, x_test)
    # -----------------

    # --- Data Balancing  ---
    print("\t - Features Selection ...")
    x_train, y_train = Preprocessing.do_balancing(_config, x_train, y_train)
    # -----------------

    # --- Model execution: SVM  ---
    print("\t - Model execution: SVM ...")
    svm.svm(x_train, y_train, x_test, y_test)
    # -----------------


if __name__ == '__main__':
    run()
