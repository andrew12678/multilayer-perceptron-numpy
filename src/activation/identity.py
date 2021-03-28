import numpy as np
from .activation import Activation
from copy import deepcopy


class Identity(Activation):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray):
        """
        Computes the Identity activation function on a numpy array
        Args:
            x (np.ndarray): The array to be applied to Identity
        """
        self.input = x
        return deepcopy(x)

    def backward(self, upstream_grad: np.ndarray):
        """
        Args:
            upstream_grad (np.ndarray):
        Computes the local gradient, combines with upstream and passes onto to downstream
        """
        return upstream_grad

    def derivative(self, x):
        return 1
