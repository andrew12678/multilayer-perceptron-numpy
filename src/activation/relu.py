import numpy as np
from .activation import Activation


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray):
        """
        Computes the ReLU activation function on a numpy array
        Args:
            x (np.ndarray): The array to be applied to ReLU
        """
        # Computes and stores for backward pass
        self.input = x
        return np.maximum(x, 0)

    def backward(self, upstream_grad: np.ndarray):
        """
        Args:
            upstream_grad (np.ndarray):
        Computes the local gradient, combines with upstream and passes onto to downstream
        Returns:

        """
        # dJ/dnet = dJ/dz * dz/dnet = dJ/dz * deriv_relu(net)
        return upstream_grad * self.derivative(self.input)

    def derivative(self, x: np.ndarray):
        """
        Computes the derivative of the RelU for an array
        Args:
            x (np.ndarray): the numpy array for which we compute the derivative on

        """
        return x > 0
