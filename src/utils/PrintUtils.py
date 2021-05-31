import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


def print_init_stats(name, df):
    print("Shape of '", name, "':\n", df.shape)
    print("Head of '", name, "':\n", df.head())

    has_nan_values = df.isnull().values.any()
    print("Has NaN '", name, "':\n", has_nan_values)
    if has_nan_values:
        #df.fillna(df.mean(), inplace=True)
        df.fillna(method='ffill', inplace=True)
        df = df.reset_index(drop=True)
        print("after filter has nan:", df.isnull().values.any())

    activities = df["activity"].unique()
    users = len(df["user"].unique())
    print("qta subjects in '", name, "':\n", users)


# --- Plot functions  ---

def plot_count_per_subject(df):
    plt.figure(figsize=(15, 8))
    plt.title('Data distribution per user')
    sns.countplot(x='user', data=df)
    plt.show()


def plot_samplings_per_class(df):
    plt.figure(figsize=(12, 8))
    plt.title('Number of sampling per class')
    sns.countplot(x='activity', data=df)
    plt.show()


def plot_sampling_per_class_per_user(df):
    plt.figure(figsize=(12, 8))
    plt.title('Number of sampling per class collected by users')
    sns.countplot(hue = 'activity', x='user', data = df)
    plt.show()


def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-acc', 'y-acc', 'z-acc']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

    plt.show()

# -----------------
