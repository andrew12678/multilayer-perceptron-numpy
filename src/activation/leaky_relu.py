import numpy as np
from .activation import Activation


class LeakyReLU(Activation):
    def __init__(self, c: int = 0.05):
        """
        Accepts an argument for the slope of the leaky ReLU for negative values
        Args:
            c ():
        """
        super().__init__()
        self.input = None
        self.c = c

    def forward(self, x: np.ndarray):
        """
        Computes the Leaky ReLU activation function on a numpy array
        Args:
            x (np.ndarray): The array to be applied to ReLU
        """
        # Computes and stores for backward pass
        self.input = x
        return np.where(x > 0, x, self.c * x)

    def backward(self, upstream_grad: np.ndarray):
        """
        Args:
            upstream_grad (np.ndarray):
        Computes the local gradient, combines with upstream and passes onto to downstream
        Returns:

        """
        # dJ/dnet = dJ/dz * dz/dnet = dJ/dz * deriv_lrelu(net)
        return upstream_grad * self.derivative(self.input)

    def derivative(self, x: np.ndarray):
        """
        Computes the derivative of the RelU for an array
        Args:
            x (np.ndarray): the numpy array for which we compute the derivative on

        """
        return np.where(x >= 0, 1, self.c)
