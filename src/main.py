import configparser

from classifiers import MLClassifiers
from utils import FileUtils, Preprocessing, PrintUtils, DataCleaning

_config = configparser.ConfigParser()
_config.read('config.ini')


def run():

    # --- Load dataset  ---
    print("\t - Load Dataset ...")
    df_data = FileUtils.load_dataset(_config)
    # -----------------

    if _config['DATASET']['dataset'] != 'local_preprocessed':
        # --- Print init stats  ---
        print("\t - Generating stats ...")
        if _config['INIT']['verbose'] == "True":
            PrintUtils.print_init_stats(_config['DATASET']['dataset'], df_data)

        generate_stats = _config['INIT']['generate_stats']
        if generate_stats == "True":
            PrintUtils.plot_count_per_subject(df_data)
            PrintUtils.plot_samplings_per_class(df_data)
            PrintUtils.plot_sampling_per_class_per_user(df_data)
            PrintUtils.plot_activity("Sitting", df_data)
            PrintUtils.plot_activity("Walking", df_data)
            PrintUtils.plot_activity("Jogging", df_data)
        # -----------------

        # --- Data Cleaning  ---
        print("\t - Data cleaning (fix NaN values) ...")
        df_data = DataCleaning.handle_nan_values(_config, df_data)
        # -----------------

        # --- Data Treatment  ---
        print("\t - Data Treatment (feat extraction) ...")
        df_data = Preprocessing.feat_extraction(_config, df_data)

        print("\t - Save df_feature ...")
        FileUtils.save_dataset(df_data)
        # -----------------

    # --- Print init stats  ---
    print("\t - Generating stats ...")
    if _config['INIT']['verbose'] == "True":
        PrintUtils.print_init_stats(_config['DATASET']['dataset'], df_data)
    # -----------------

    # --- Encode labels  ---
    print("\t - Encoding labels ...")
    df_data = Preprocessing.encode_labels(df_data)
    # -----------------

    # --- Train/Test Split  ---
    print("\t - Train/Test Split ...")
    x_train, x_test, y_train, y_test = Preprocessing.do_train_test_split(_config, df_data)
    # -----------------

    if _config['INIT']['verbose'] == "True":
        PrintUtils.plot_tsne(x_train, y_train)


    # --- Features Selection  ---
    print("\t - Features Selection ...")
    x_train, x_test = Preprocessing.do_features_selection(_config, x_train, y_train, x_test)
    # -----------------

    # --- Normalization  ---
    print("\t - Normalization ...")
    x_train, x_test = Preprocessing.scale_dataset(_config, x_train, x_test)
    # -----------------

    if _config['INIT']['verbose'] == "True":
        PrintUtils.plot_tsne(x_train, y_train)

    # --- Data Balancing  ---
    print("\t - Data Balancing ...")
    x_train, y_train = Preprocessing.do_balancing(_config, x_train, y_train)
    # -----------------

    # --- Model execution: ML  ---
    print("\t - Model execution: ML ...")
    MLClassifiers.run_classifiers(_config, x_train, y_train, x_test, y_test)
    # -----------------


if __name__ == '__main__':
    run()




