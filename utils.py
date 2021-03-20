import numpy as np
import os
from sklearn.model_selection import StratifiedKFold


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


def create_kfold_stratfied_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
):
    """
    Combines the given train-test dataset and creates the k-fold cross validation folds for them
    Args:
        X_train (np.ndarray): the training features
        y_train (np.ndarray): the training labels
        X_test (np.ndarray): the testing features
        y_test (np.ndarray): the testing labels
        k (int): the number of cross validation folds

    Returns:
        A list where containing the k-folds (could be optimised later to only return the indices)
    """
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(X_all, y_all)
    splits = []
    for train_index, test_index in skf.split(X_all, y_all):
        splits.append(
            (
                X_all[train_index],
                y_all[train_index],
                X_all[test_index],
                y_all[test_index],
            )
        )
    return splits
