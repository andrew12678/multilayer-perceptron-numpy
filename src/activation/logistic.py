import numpy as np
from .activation import Activation
from ..utils.calculations import sigmoid


class Logistic(Activation):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray):
        """
        Computes the Logistic activation function on a numpy array
        Args:
            x (np.ndarray): The array to be applied to Logistic/Sigmoid
        """
        self.input = x
        return sigmoid(x)

    def backward(self, upstream_grad: np.ndarray):
        """
        Args:
            upstream_grad (np.ndarray):
        Computes the local gradient, combines with upstream and passes onto to downstream
        """
        return upstream_grad * self.derivative(self.input)

    def derivative(self, x):
        return sigmoid(x) / (1.0 - sigmoid(x))
