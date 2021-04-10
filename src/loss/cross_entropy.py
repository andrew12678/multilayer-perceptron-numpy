import numpy as np
from copy import deepcopy
from .loss import Loss
from ..utils.calculations import softmax


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        self.delta = None

    # Calculate forward-pass output loss
    def forward(self, y: np.ndarray, y_hat: np.ndarray):

        """
        Apply softmax to y_hat to make into a probability distribution and then calculate the cross-entropy loss of
        the mini-batch
        Args:
            y (np.ndarray): A numpy array of shape (N,C) which contains the one-hotted labels for the mini-batch
            y_hat (np.ndarray): A numpy array of shape (N,C) which contains the outputs per class for mini-batch

        """

        # Compute softmax on y_hat
        softmax_y_hat = softmax(y_hat)

        # Store delta for later
        self.delta = deepcopy(softmax_y_hat)
        self.delta -= y

        # Return mean loss for sample batch
        return -(y * np.log(softmax_y_hat + 1e-9)).mean()

    # Compute gradient for backwards propagation
    def backward(self, upstream_grad: int = 1):

        """
        Computes the local gradient and multiplies it with the upstream gradient to pass onto downstream

        Args:
            upstream_grad (int): upstream gradient from the previous layer (in this case it's just 1)

        Returns:

        """

        # For batched CE, dJ/dy_hat = dJ/dJ * (-1/N) * (softmax(y_hat) - y)
        return upstream_grad * (1 / self.delta.shape[0]) * self.delta
