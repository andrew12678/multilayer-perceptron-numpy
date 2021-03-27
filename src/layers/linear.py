from .layer import Layer
import numpy as np


class Linear(Layer):
    def __init__(self, n_in: int, n_out: int):
        """
        Sets up a linear layer taking in n_in inputs and outputs n_out outputs
        Args:
            n_in (int): number of inputs to take
            n_out (int): number of outputs to take
        """
        super().__init__()

        # Randomly initialise weights (implement Xavier/He later)
        self.weights = np.random.uniform(
            low=-np.sqrt(6.0 / (n_in + n_out)),
            high=np.sqrt(6.0 / (n_in + n_out)),
            size=(n_in, n_out),
        )

        # A bias for each output
        self.biases = np.zeros(
            n_out,
        )

        # Set gradients as the size of weight
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)

        # To store the input later
        self.input = None
        self.output = None

    def forward(self, X: np.ndarray):
        """
        Computes the forward pass of the layer
        Args:
            X (np.ndarray): the input array

        Returns:

        """
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, upstream_grad: np.ndarray):
        """
        Computes the backward pass of the layer
        Args:
            upstream_grad (np.ndarray): the gradient from the upstream

        Returns:

        """
        self.grad_W = np.dot(self.input.T, upstream_grad)
        self.grad_b = upstream_grad.sum(axis=0)
        return np.dot(upstream_grad, self.weights.T)

    def zero_grad(self):
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)
