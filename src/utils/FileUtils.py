import os
import pandas as pd


def load_dataset(source_file, delimiter):
    # TODO: Doc
    columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
    df_dataset = pd.read_csv(source_file, sep=delimiter, header=None, names=columns)

    df_dataset = df_dataset.replace({';': ''}, regex=True)
    df_dataset['x-acc'] = df_dataset['x-acc'].astype(float)
    df_dataset['y-acc'] = df_dataset['y-acc'].astype(float)
    df_dataset['z-acc'] = df_dataset['z-acc'].astype(float)
    return df_dataset
