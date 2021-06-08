import pandas as pd
import os


def load_dataset(_config):
    """
    read local database
    """

    dataset_type = _config['DATASET']['dataset']
    file_name = ""
    if dataset_type == 'wisdm1':
        file_name = "./data/WISDM_ar_v1.1_raw.txt"
    if dataset_type == 'wisdm2':
        file_name = "./data/WISDM_at_v2.0_raw.txt"
    if dataset_type == "wisdm3":
        dir = "./data/wisdmv3/"
        df_dataset = pd.DataFrame()

        for filename in os.listdir(dir):
            columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
            tmp_df = pd.read_csv(dir + filename, sep=",", header=None, names=columns)
            df_dataset = df_dataset.append(tmp_df, ignore_index=True)


        df_dataset = df_dataset.replace({';': ''}, regex=True)
        df_dataset['x-acc'] = df_dataset['x-acc'].astype(float)
        df_dataset['y-acc'] = df_dataset['y-acc'].astype(float)
        df_dataset['z-acc'] = df_dataset['z-acc'].astype(float)
        return df_dataset
    if dataset_type == "local_preprocessed":
        return pd.read_csv("./data/df_feature.csv", sep=",")

    columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
    df_dataset = pd.read_csv(file_name, sep=",", header=None, names=columns)

    df_dataset = df_dataset.replace({';': ''}, regex=True)
    df_dataset['x-acc'] = df_dataset['x-acc'].astype(float)
    df_dataset['y-acc'] = df_dataset['y-acc'].astype(float)
    df_dataset['z-acc'] = df_dataset['z-acc'].astype(float)

    return df_dataset


def save_dataset(df):
    """
    Save preprocessed database locally
    """

    df.to_csv('./data/df_feature.csv', index=False)
