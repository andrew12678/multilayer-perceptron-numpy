from .optimiser import Optimiser
from typing import List
from ..layers.linear import Layer
import numpy as np


class Adagrad(Optimiser):
    def __init__(
        self, layers: List[Layer], learning_rate: float, weight_decay: float = 0
    ):
        """
        Initialise the Adagrad with parameters.
        Args:
            layers (List[Layer]): List of parameters holding Layer objects (either BN or Linear)
            learning_rate (float): the learning rate of the optimiser
            weight_decay (float): the decay factor applied to the weights before an update.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gradient_sum_squares = [
            [np.zeros(layer.grad_W.shape), np.zeros(layer.grad_B.shape)]
            for layer in layers
        ]  # Set this to 0 initially since all gradients are 0
        self.epsilon = 1e-9

    def step(self):

        for idx, layer in enumerate(self.layers):
            # Update the gradient sum of squares for the layer
            self.gradient_sum_squares[idx][0] += layer.grad_W ** 2
            self.gradient_sum_squares[idx][1] += layer.grad_B ** 2

            # Calculate the modified learning rates for both parameters
            modified_learning_rate_W = self.lr / np.sqrt(
                self.epsilon + self.gradient_sum_squares[idx][0]
            )
            modified_learning_rate_B = self.lr / np.sqrt(
                self.epsilon + self.gradient_sum_squares[idx][1]
            )

            # Update weights
            layer.weights = (1 - self.weight_decay) * (
                layer.weights - modified_learning_rate_W * layer.grad_W
            )

            # Update biases
            layer.biases -= modified_learning_rate_B * layer.grad_B
