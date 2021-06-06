import math

import numpy as np
import sensormotion as sm


def handle_nan_values(_config, df):
    """
    Fix NaN values during loading phase of the dataset.
    The available technique is: forward fill, backward fill, mean fill.
    """

    cleaning_method = _config['DATA_CLEANING']['clean_method']

    has_nan_values = df.isnull().values.any()
    if not has_nan_values:
        print("\t\t- Dataset don't contains any NaN values.")
        print("\t\t- Skipping")
        return df

    df.replace([math.inf, -math.inf], np.nan, inplace=True)
    if cleaning_method == 'mean':
        df.fillna(df.mean(), inplace=True)
    else:
        df.fillna(method=cleaning_method, inplace=True)

    df = df.reset_index(drop=True)

    if not df.isnull().values.any():
        print("\t\t- After apply '", cleaning_method, "' the dataset don't contains NaN values.")

    return df
