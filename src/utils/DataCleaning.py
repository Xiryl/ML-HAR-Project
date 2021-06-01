import math

import numpy as np
import sensormotion as sm


def apply_filter(_config, df):
    filter = _config['DATA_CLEANING']['filter']
    sampling_frequency = int(_config['DATA_REPRESENTATION']['sampling_frequency'])
    filter_order = int(_config['DATA_CLEANING']['filter_order'])
    filter_high_cut = int(_config['DATA_CLEANING']['filter_high_cut'])
    filter_low_cut = int(_config['DATA_CLEANING']['filter_low_cut'])

    if filter == 'none':
        return df

    if filter == 'lowpass':
        cut_off = filter_low_cut

    if filter == 'highpass':
        cut_off = filter_high_cut

    b, a = sm.signal.build_filter(frequency=20,
                                  sample_rate=sampling_frequency,
                                  filter_type=filter,
                                  filter_order=filter_order)

    # df.replace([math.inf, -math.inf], np.nan, inplace=True)
    # df = df.reset_index(drop=True)

    df['x-acc'] = sm.signal.filter_signal(b, a, signal=df['x-acc'].values)
    df['y-acc'] = sm.signal.filter_signal(b, a, signal=df['y-acc'].values)
    df['z-acc'] = sm.signal.filter_signal(b, a, signal=df['z-acc'].values)

    return df


def handle_nan_values(_config, df):
    cleaning_method = _config['DATA_CLEANING']['clean_method']

    has_nan_values = df.isnull().values.any()
    if not has_nan_values:
        print("\t\t- Dataset don't have NaN values.")

    df.replace([math.inf, -math.inf], np.nan, inplace=True)
    if cleaning_method == 'mean':
        df.fillna(df.mean(), inplace=True)
    else:
        df.fillna(method=cleaning_method, inplace=True)

    df = df.reset_index(drop=True)
    if not df.isnull().values.any():
        print("\t\t- After apply '",cleaning_method,"' the dataset don't have NaN values.")

    return df