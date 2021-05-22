import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def load_dataset(train_data, test_data):
    """
    @brief Load csv file from local source

    :param train_data: path for train data .csv file
    :param test_data: path for test data .csv file
    :return: df_train, df_test (dataframe)
    """
    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)
    return df_train, df_test
