import os
import pandas as pd


def load_dataset(_config):
    dataset_type = _config['DATASET']['dataset']
    file_name = ""
    if dataset_type == 'wisdm1':
         file_name = "./data/WISDM_ar_v1.1_raw.txt"
    if dataset_type == 'wisdm2':
        file_name = "./data/WISDM_at_v2.0_raw.txt"

    columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
    df_dataset = pd.read_csv(file_name, sep=",", header=None, names=columns)

    df_dataset = df_dataset.replace({';': ''}, regex=True)
    df_dataset['x-acc'] = df_dataset['x-acc'].astype(float)
    df_dataset['y-acc'] = df_dataset['y-acc'].astype(float)
    df_dataset['z-acc'] = df_dataset['z-acc'].astype(float)
    return df_dataset
