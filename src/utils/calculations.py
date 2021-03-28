import numpy as np


def softmax(x: np.ndarray):
    """
    Computes a numerically stable softmax by first shifting the distribution for each sample
    Args:
        x (np.ndarray): A numpy array of shape (N,C)

    Returns:

    """
    # Shift each row for numerically stability
    shift = x - x.max(axis=1, keepdims=True)

    # Compute softmax
    exp_shift = np.exp(shift)
    softmax_x = exp_shift / exp_shift.sum(axis=1, keepdims=True)
    return softmax_x


def sigmoid(x: np.ndarray):
    """
    Computes the sigmoid of a numpy array
    Args:
        x (np.ndarray): the numpy array to be provided to the sigmoid function

    Returns:

    """
    return 1.0 / (1.0 + np.exp(-x))
