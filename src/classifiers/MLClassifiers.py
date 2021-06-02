from classifiers import svm, knn


def run_classifiers(_config, x_train, y_train, x_test, y_test):
    cfg_svm = _config['ML_CLF_EXECUTION']['svm']
    cfg_svm_gs = _config['ML_CLF_EXECUTION']['svm_gs']
    cfg_knn = _config['ML_CLF_EXECUTION']['knn']
    cfg_knn_gs = _config['ML_CLF_EXECUTION']['knn_gs']

    if cfg_svm == 'True':
        print("\t\t - Executing: SVM ...")
        cfg_kernel = _config['SVM']['kernel']
        cfg_gamma = _config['SVM']['gamma']
        cfg_C = float(_config['SVM']['C'])
        cfg_max_iter = int(_config['SVM']['max_iter'])
        svm.svm(x_train, y_train, x_test, y_test, cfg_kernel, cfg_gamma, cfg_C, cfg_max_iter)

    if cfg_svm_gs == 'True':
        print("\t\t - Executing: SVM GS ...")
        svm.svm_gs(x_train, y_train, x_test, y_test)

    if cfg_knn == 'True':
        print("\t\t - Executing: KNN ...")
        cfg_n_neighbors = int(_config['KNN']['n_neighbors'])
        cfg_p = int(_config['KNN']['p'])
        cfg_metric = _config['KNN']['metric']
        knn.knn(x_train, y_train, x_test, y_test, cfg_n_neighbors, cfg_p, cfg_metric)

    if cfg_knn_gs == 'True':
        print("\t\t - Executing: KNN GS ...")
        knn.knn_gs(x_train, y_train, x_test, y_test)

    return