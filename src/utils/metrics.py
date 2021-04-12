import numpy as np


def calculate_metrics(y: np.ndarray, y_hat: np.ndarray):

    """
    Calls all metrics calculations and returns dict of all metrics
    Args:
        y (np.ndarray): the true labels for a set of samples
        y_hat (np.ndarray): the predicted labels for a set of samples
        accuracy (bool): boolean for calculating simple accuracy or not
        f1 (bool): boolean for calculating f1 macro score or not
    Returns:
        metric_dict (Dict): dictionary containing calculated metrics
    """

    # Create dictionary containing all metrics
    metrics_dict = {
        "accuracy": classification_accuracy(y, y_hat),
        "f1_macro": f1_macro(y, y_hat),
    }

    # Return metrics dictionary
    return metrics_dict


def classification_accuracy(y: np.ndarray, y_hat: np.ndarray):

    """
    Computes the simple classification accuracy between the true and predicted labels
    Args:
        y (np.ndarray): the true labels for a set of samples
        y_hat (np.ndarray): the predicted labels for a set of samples
    Returns:

    """

    # Return fraction of matching labels between predicted and true arrays
    return np.mean(y_hat == y)


def f1_macro(y: np.ndarray, y_hat: np.ndarray):

    """
    Computes the macro f1 score between the true and predicted labels
    Args:
        y (np.ndarray): the true labels for a set of samples
        y_hat (np.ndarray): the predicted labels for a set of samples
    Returns:

    """

    # Create list for storing f1 scores for each label/class
    f1_scores = []

    # Loop through all unique labels
    for current_label in np.unique(y):

        # Calculate true positives
        tp = np.sum((y == current_label) & (y_hat == current_label))

        # Calculate false positives
        fp = np.sum((y != current_label) & (y_hat == current_label))

        # Calculate false negatives
        fn = np.sum((y == current_label) & (y_hat != current_label))

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        # Calculate f1 score for current label
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        # Append f1 score to list
        f1_scores.append(f1)

    # Return mean f1 score across all labels
    return np.mean(f1_scores)
