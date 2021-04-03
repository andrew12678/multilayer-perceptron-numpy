import numpy as np
import math


def one_hot(y: np.ndarray):
    """
    One-hots an numpy array of labels
    Args:
        y (np.ndarray): numpy array of labels

    Returns:

    """

    n = y.shape[0]
    max_v = y.max()
    one_hot_array = np.zeros((n, max_v + 1))
    one_hot_array[np.arange(n), y.reshape(-1)] = 1
    return one_hot_array


def create_kfold_stratified_cross_validation(
    X_train: np.ndarray, y_train: np.ndarray, k: int = 5, seed: int = 18
):
    """
    Combines the given train-test dataset and creates the k-fold cross validation folds for them without
    sklearn
    Args:
        X_train (np.ndarray): the training features
        y_train (np.ndarray): the training labels
        k (int): the number of cross validation folds
        seed (int): the random seed for reproducibility

    Returns:
        A list where containing the k-folds (could be optimised later to only return the indices)
    """
    X_all = X_train
    y_all = y_train
    # Get the number of elements per group
    group_counts = np.bincount(y_all.flatten())

    # Gets the original indices of each group
    # e.g. [2,0,3,1,1] -> [[1],[3,4],[0],[2],[]]
    original_group_indices = np.split(
        y_all.flatten().argsort(), np.cumsum(group_counts)
    )

    # Splits the grouped indices of each element into k
    group_fold_indices = []
    for group_count in group_counts:
        group_indices = np.arange(group_count)
        np.random.seed(seed)
        np.random.shuffle(group_indices)
        group_fold_indices.append(np.array_split(group_indices, k))

    # Generate k-folds with some output as sklearn.StratifiedKFold
    splits = []
    for fold_index in range(k):
        # For each fold we consider get the relevant group indices we split into k subgroups
        # and look them up in the original
        test_indices = np.concatenate(
            [
                original_group_indices[group_index][group[fold_index]]
                for group_index, group in enumerate(group_fold_indices)
            ]
        )
        # Uses masks to allow negation
        mask = np.zeros(X_all.shape[0], dtype=bool)
        mask[test_indices] = True
        X_train = X_all[~mask]
        y_train = y_all[~mask]
        X_test = X_all[mask]
        y_test = y_all[mask]
        splits.append((X_train, y_train, X_test, y_test))

    return splits


class Batcher:
    """
    Class for generating mini-batch indices, run at the beginning of every epoch
    """

    def __init__(self, data_size: int, batch_size: int = 64):
        """
        Initialise the batch
        Args:
            data_size (int): the total number of samples in the dataset
            batch_size (int): the number of samples in each batch
        """
        self.indices = np.arange(data_size)
        self.count_batches = math.ceil(data_size / batch_size)

    def generate_batch_indices(self):
        """
        Generates a fresh set of batch indices
        Returns:
            List of numpy array
        """
        np.random.shuffle(self.indices)
        return np.array_split(self.indices, self.count_batches)
