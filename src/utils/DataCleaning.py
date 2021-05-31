import math

import numpy as np
import sensormotion as sm


def apply_filter(_config, df):
    filter = _config['DATA_CLEANING']['filter']

    if filter == 'none':
        return df

    b, a = sm.signal.build_filter(frequency=0.1,
                                  sample_rate=20,
                                  filter_type='lowpass',
                                  filter_order=4)

    df.replace([math.inf, -math.inf], np.nan, inplace=True)
    df = df.reset_index(drop=True)

    df['x-acc'] = sm.signal.filter_signal(b, a, signal=df['x-acc'].values)
    df['y-acc'] = sm.signal.filter_signal(b, a, signal=df['y-acc'].values)
    df['z-acc'] = sm.signal.filter_signal(b, a, signal=df['z-acc'].values)

    return df

def handle_nan_values(_config, df):
    cleaning_method = _config['DATA_CLEANING']['clean_method']

    has_nan_values = df.isnull().values.any()
    if not has_nan_values:
        print("\t\t- Dataset don't have NaN values.")

    df.fillna(method=cleaning_method, inplace=True)
    df = df.reset_index(drop=True)
    if not df.isnull().values.any():
        print("\t\t- After apply '",cleaning_method,"' the dataset don't have NaN values.")

    return df