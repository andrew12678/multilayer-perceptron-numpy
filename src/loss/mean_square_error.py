import numpy as np

from .loss import Loss


class MeanSquaredErrorLoss(Loss):
    def __init__(self):
        super().__init__()
        self.delta = None

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """
        Calculate the MSE loss of the mini-batch
        Args:
            y (np.ndarray): A numpy array of shape (N,C) which contains the one-hotted labels for the mini-batch
            y_hat (np.ndarray): A numpy array of shape (N,C) which contains the outputs per class for mini-batch

        Returns:

        """
        # Storing (y - y_hat) for later re-use in backpropagation
        self.delta = y - y_hat
        return 0.5 * (self.delta ** 2).mean()

    def backward(self, upstream_grad: int = 1):
        """
        Computes the local gradient and multiplies it with the upstream gradient to pass onto downstream
        Args:
            upstream_grad (int): upstream gradient from the previous layer (in this case it's just 1)

        Returns:

        """
        # For batched MSE, dJ/dy_hat = dJ/dJ * (-1/N) * (y-y_hat)
        return upstream_grad * (-1 / self.delta.shape[0]) * self.delta
