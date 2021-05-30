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

    if generate_stats == "True":
        print("Executing using [ generate stats ] as init phase.")
    else:
        print("Executing using [ no generate stats ] as init phase.")

    if filter == "none":
        print("Executing using [ no FFT filter ] as data cleaning type.")
    else:
        print("Executing using [ FFT filter: low ] as data cleaning type.")

    #print("Executing using [", data_treatment, "] as data treatment type.")
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

    # --- Print init stats  ---
    print("\t - Generating stats ...")
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
    df_wisdm_v1_filtered = DataCleaning.apply_filter(_config, df_data)
    df_data = df_wisdm_v1_filtered
    # -----------------

    # --- Encode labels  ---
    print("\t - Encoding labels ...")
    df_wisdm_v1_encoded = Preprocessing.encode_labels(df_data)
    df_data = df_wisdm_v1_encoded
    # -----------------


if __name__ == '__main__':
    run()
