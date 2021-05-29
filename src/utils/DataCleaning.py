import math

import numpy as np
import sensormotion as sm


def apply_filter(df):
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