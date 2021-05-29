import os
import pandas as pd


def load_dataset(source_file, delimiter):
    # TODO: Doc
    columns = ['user', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
    df_dataset = pd.read_csv(source_file, sep=delimiter, header=None, names=columns)
    return df_dataset
