from utils import FileUtils, Preprocessing, PrintUtils, DataCleaning
from classifiers import svm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


def run(data_treatment, apply_fft, generate_stats):
    # --- Print running configuration  ---
    if generate_stats == "True":
        print("Executing using [ generate stats ] as init phase.")
    else:
        print("Executing using [ no generate stats ] as init phase.")

    if apply_fft == "True":
        print("Executing using [ FFT filter ] as data cleaning type.")
    else:
        print("Executing using [ no FFT filter ] as data cleaning type.")

    print("Executing using [", data_treatment, "] as data treatment type.")
    # -----------------

    print("-----------------")

    # --- Load datasets  ---
    print("\t - Load Dataset ...")
    df_wisdm_v1 = FileUtils.load_dataset("./data/WISDM_ar_v1.1_raw.txt", ",")
    # df_wisdm_v1 = FileUtils.load_dataset("./data/WISDM_at_v2.0_raw.txt", ",")
    # -----------------

    # --- Print init stats  ---
    print("\t - Generating stats ...")
    PrintUtils.print_init_stats("WISDM_v1", df_wisdm_v1)
    PrintUtils.plot_count_per_subject(df_wisdm_v1)
    PrintUtils.plot_samplings_per_class(df_wisdm_v1)
    PrintUtils.plot_sampling_per_class_per_user(df_wisdm_v1)
    PrintUtils.plot_activity("Sitting", df_wisdm_v1)
    PrintUtils.plot_activity("Walking", df_wisdm_v1)
    PrintUtils.plot_activity("Jogging", df_wisdm_v1)
    # -----------------

    # --- Data FFT filtering  ---
    print("\t - Data filtering (noise removal) ...")
    if apply_fft:
        df_wisdm_v1_filtered = DataCleaning.apply_filter(df_wisdm_v1)
        df_wisdm_v1 = df_wisdm_v1_filtered
    # -----------------

    # --- Encode labels  ---
    print("\t - Encoding labels ...")
    df_wisdm_v1_encoded = Preprocessing.encode_labels(df_wisdm_v1)
    df_wisdm_v1 = df_wisdm_v1_encoded
    # -----------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML HAR Project.')
    parser.add_argument('--data_treatment',
                        default="feat_extraction",
                        help='Choose the data treatment type (segmentation, feat_extraction')
    parser.add_argument('--apply_fft',
                        default=False,
                        help='If True, it apply FFT filter to the data')
    parser.add_argument('--generate_stats',
                        default=False,
                        help='If True, it generate some init stats and plots')

    args = parser.parse_args()
    data_treatment = args.data_treatment
    apply_fft = args.apply_fft
    generate_stats = args.generate_stats

    run(data_treatment, apply_fft, generate_stats)
