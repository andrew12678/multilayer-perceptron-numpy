import numpy as np
import os


def load_directory(path: str = "data/"):
    """
    Loads the datasets in a directory and returns a tuple of Numpy arrays
    Args:
        path (str): path to the datasets

    Returns:
        A tuple of numpy arrays containing the training/testing data
    """
    X_train = np.load(os.path.join(path, "train_data.npy"))
    y_train = np.load(os.path.join(path, "train_label.npy"))
    X_test = np.load(os.path.join(path, "test_data.npy"))
    y_test = np.load(os.path.join(path, "test_label.npy"))
    return X_train, y_train, X_test, y_test
