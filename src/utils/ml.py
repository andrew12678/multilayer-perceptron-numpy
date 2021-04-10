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


def create_stratified_kfolds(X_train: np.ndarray, y_train: np.ndarray, k: int = 5):
    """
    Creates k-folds cross validation data given X_train and y_train.
    Args:
        X_train (np.ndarray): the training features
        y_train (np.ndarray): the training labels
        k (int): the number of cross validation folds

    Returns:
        A list containing the data for k-folds
    """
    X_all = X_train
    y_all = y_train
    # Get the number of elements per class
    group_counts = np.bincount(y_all.flatten())

    # Get a list of indices representing each class, in order of class value.
    # e.g. if y_all = [0,2,1,1,2] then original_group_indices = [[0], [2,3], [1,4]]
    original_group_indices = np.split(
        y_all.flatten().argsort(), np.cumsum(group_counts)
    )

    # Splits the grouped indices of each element into k subgroups.
    group_fold_indices = []
    for group_count in group_counts:
        group_indices = np.arange(group_count)
        np.random.shuffle(group_indices)
        group_fold_indices.append(np.array_split(group_indices, k))

    # Generate k-folds with same output as sklearn.StratifiedKFold
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


def initialise_weights(n_in: int, n_out: int, activation_fn: str):
    """
    Get initialisations for a weight matrix of a network.
    Args:
        n_in (int): number of inputs to take for each neuron in the layer
        n_out (int): number of outputs/neurons
        activation_fn (str): type of activation function used for weight initialisation (he, xavier, xavier4)
    """
    # Check if Xavier initialisation (Sigmoid/Logistic)
    if activation_fn == "logistic":

        # Randomly initialise weights according to Xavier
        weights = np.random.uniform(
            low=-np.sqrt(6.0 / (n_in + n_out)),
            high=np.sqrt(6.0 / (n_in + n_out)),
            size=(n_in, n_out),
        )

    # Check if Xavier initialisation (Tanh)
    elif activation_fn == "tanh":

        # Initialise Xavier weights and multiply by 4
        weights = (
            np.random.uniform(
                low=-np.sqrt(6.0 / (n_in + n_out)),
                high=np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out),
            )
            * 4
        )

    # Check if He initialisation (ReLU & LReLU)
    elif activation_fn == "relu" or activation_fn == "leaky_relu":

        # Initialise weights using Kaiming uniform distribution
        weights = np.random.uniform(
            low=-np.sqrt(6.0 / n_in), high=np.sqrt(6.0 / n_in), size=(n_in, n_out)
        )

    # Check if activation is not defined (REVIEW THIS)
    elif activation_fn == None:

        # Initialise weights using Kaiming uniform distribution
        weights = np.random.uniform(
            low=-np.sqrt(6.0 / n_in), high=np.sqrt(6.0 / n_in), size=(n_in, n_out)
        )

    # Check if unknown activation function entered
    else:
        # Raise exception indicating that initialisation method is unknown
        raise ValueError(
            f"'{activation_fn}' is not a recognised activation function for weight initialisation."
        )
    return weights


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
