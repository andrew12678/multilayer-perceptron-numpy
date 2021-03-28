import numpy as np
from .activation import Activation


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray):
        """
        Computes the Tanh activation function on a numpy array
        Args:
            x (np.ndarray): The array to be applied to Tanh
        """
        self.input = x
        return np.tanh(x)

    def backward(self, upstream_grad: np.ndarray):
        """
        Args:
            upstream_grad (np.ndarray):
        Computes the local gradient, combines with upstream and passes onto to downstream
        """
        return upstream_grad * self.derivative(self.input)

    def derivative(self, x):
        return 1.0 - np.tanh(x) ** 2
