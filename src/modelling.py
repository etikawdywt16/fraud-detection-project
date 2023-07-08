from utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load configuration file
    config = load_config()

    # load dataset
    X_train_clean_rus = load_dataset('X_train_clean_rus.pkl')
    y_train_clean_rus = load_dataset('y_train_clean_rus.pkl')

    X_test_clean = load_dataset('X_test_clean.pkl')
    y_test_clean = load_dataset('y_test_clean.pkl')

    # training model
    final_model = train_model(X_train_clean_rus, y_train_clean_rus)

    # evaluation model
    evaluation_model(final_model, X_test_clean, y_test_clean)

    # dump model
    dump_model(final_model, 'Decision_Tree_Classifier.pkl')