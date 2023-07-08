from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import *


if __name__ == '__main__':
    # load configuration file
    config = load_config('../config/config.yaml')

    # load dataset
    X_train = load_dataset('X_train.pkl', config = config)
    y_train = load_dataset('y_train.pkl', config = config)
    X_test = load_dataset('X_test.pkl', config = config)
    y_test = load_dataset('y_test.pkl', config = config)

    # preprocess features dataset
    X_train_preprocess = preprocess_input(X_train, config = config)
    X_test_preprocess = preprocess_input(X_test, config = config)
    
    # encoding features and target dataset
    X_train_ohe = ohe_input(X_train_preprocess, config = config)
    y_train_ohe = ohe_output(y_train)

    X_test_ohe = ohe_input(X_test_preprocess, config = config)
    y_test_ohe = ohe_output(y_test)

    # clean features
    X_train_clean = clean_input(X_train_preprocess, X_train_ohe)
    y_train_clean = y_train_ohe

    X_test_clean = clean_input(X_test_preprocess, X_test_ohe)
    y_test_clean = y_test_ohe

    # dump file
    dump_dataset(X_train_clean, 'X_train_clean.pkl', config = config)
    dump_dataset(y_train_clean, 'y_train_clean.pkl', config = config)
    dump_dataset(X_test_clean, 'X_test_clean.pkl', config = config)
    dump_dataset(y_test_clean, 'y_test_clean.pkl', config = config)

    # resampling train dataset
    # undersampling
    X_train_clean_rus, y_train_clean_rus = resampling(X_train_clean, y_train_clean, RandomUnderSampler, config = config)
    
    # oversampling
    X_train_clean_ros, y_train_clean_ros = resampling(X_train_clean, y_train_clean, RandomOverSampler, config = config)

    # dump file
    dump_dataset(X_train_clean_rus, 'X_train_clean_rus.pkl', config = config)
    dump_dataset(y_train_clean_rus, 'y_train_clean_rus.pkl', config = config)
    dump_dataset(X_train_clean_ros, 'X_train_clean_ros.pkl', config = config)
    dump_dataset(y_train_clean_ros, 'X_train_clean_ros.pkl', config = config)















