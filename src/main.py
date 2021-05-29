from utils import FileUtils, Preprocessing, PrintUtils
from classifiers import svm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


def run(data_treatment):
    print("Executing using [",data_treatment,"] data treatment type.")

    df_wisdm_v1 = FileUtils.load_dataset("./data/WISDM_ar_v1.1_raw.txt", ",")
    #df_wisdm_v2 = FileUtils.load_dataset("./data/WISDM_at_v2.0_raw.txt", ",")
    PrintUtils.print_init_stats("WISDM_v1", df_wisdm_v1)
    PrintUtils.plot_count_per_subject(df_wisdm_v1)
    PrintUtils.plot_samplings_per_class(df_wisdm_v1)
    PrintUtils.plot_sampling_per_class_per_user(df_wisdm_v1)
    PrintUtils.plot_activity("Sitting", df_wisdm_v1)
    PrintUtils.plot_activity("Walking", df_wisdm_v1)
    PrintUtils.plot_activity("Jogging", df_wisdm_v1)
    PrintUtils.plot_class_distribution(df_wisdm_v1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML HAR Project.')
    parser.add_argument('--data_treatment',
                        default="feat_extraction",
                        help='Choose the data treatment type (segmentation, feat_extraction')

    args = parser.parse_args()
    data_treatment = args.data_treatment
    run(data_treatment)

