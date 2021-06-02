import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def print_init_stats(name, df):
    """
    Print stats about dataset
    """
    print("\t\t- Shape of '", name, "':", df.shape)

    has_nan_values = df.isnull().values.any()
    print("\t\t- Has NaN '", name, "':", has_nan_values)

    activities = df["activity"].unique()
    users = len(df["user"].unique())
    print("\t\t- Qta subjects in '", name, "':", users)
    print("\t\t- Qta activities in '", name, "':", activities)


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
    sns.countplot(hue='activity', x='user', data=df)
    plt.show()


def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-acc', 'y-acc', 'z-acc']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

    plt.show()

# -----------------
