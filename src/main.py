from utils import FileUtils, Preprocessing


def run():
    df_train, df_test = FileUtils.load_dataset('./data/train.csv', './data/test.csv')

    # split data
    y_train = df_train.Activity
    x_train = df_train.drop(['subject', 'Activity'], axis=1)
    y_test = df_test.Activity
    x_test = df_test.drop(['subject', 'Activity'], axis=1)

    # encode labels
    y_train_encoded = Preprocessing.encode_labels(y_train)
    y_test_encoded = Preprocessing.encode_labels(y_test)

    # scale data
    x_train_s, x_test_s = Preprocessing.scale_dataset(x_train, x_test, scaler_type='standardscaler')
    print(y_train_encoded)


if __name__ == '__main__':
    run()

