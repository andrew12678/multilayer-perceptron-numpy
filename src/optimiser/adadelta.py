from .optimiser import Optimiser
from typing import List
from ..layers.linear import Layer
import numpy as np


class Adadelta(Optimiser):
    def __init__(
        self,
        layers: List[Layer],
        learning_rate: float,
        weight_decay: float,
        exponential_decay: float,
    ):
        """
        Initialise the Adadelta with parameters.
        Args:
            layers (List[Layer]): List of parameters holding Layer objects (either BN or Linear)
            learning_rate (float): the learning rate of the optimiser
            weight_decay (float): the decay factor applied to the weights before an update.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.exponential_decay = exponential_decay
        self.weight_decay = weight_decay
        self.gradient_squares_moving_average = [
            [np.zeros(layer.grad_W.shape), np.zeros(layer.grad_b.shape)]
            for layer in layers
        ]  # Set this to 0 initially since all gradients are 0

        self.previous_update = [
            [np.zeros(layer.grad_W.shape), np.zeros(layer.grad_b.shape)]
            for layer in layers
        ]  # Set this to 0 initially since all previous gradients are 0
        self.epsilon = 1e-6

    def step(self):

        for idx, layer in enumerate(self.layers):
            # Compute the new gradient square MA with decay
            new_grad_W = self.exponential_decay * self.gradient_squares_moving_average[
                idx
            ][0] + (1 - self.exponential_decay) * (layer.grad_W ** 2)
            new_grad_b = self.exponential_decay * self.gradient_squares_moving_average[
                idx
            ][1] + (1 - self.exponential_decay) * (layer.grad_b ** 2)

            # Update for next time
            self.gradient_squares_moving_average[idx][0] = new_grad_W
            self.gradient_squares_moving_average[idx][1] = new_grad_b

            # Compute the RMS of the gradient square MA
            rms_grad_W = np.sqrt(new_grad_W + self.epsilon)
            rms_grad_b = np.sqrt(new_grad_b + self.epsilon)

            # Compute the unit corrects RMS
            rms_previous_update_W = np.sqrt(self.previous_update[idx][0] + self.epsilon)
            rms_previous_update_B = np.sqrt(self.previous_update[idx][1] + self.epsilon)

            # Compute the updates
            update_W = rms_previous_update_W * layer.grad_W / rms_grad_W
            update_B = rms_previous_update_B * layer.grad_b / rms_grad_b

            # Decay on the previous update
            self.previous_update[idx][
                0
            ] = self.exponential_decay * self.previous_update[idx][0] + (
                1 - self.exponential_decay
            ) * (
                update_W ** 2
            )

            self.previous_update[idx][
                1
            ] = self.exponential_decay * self.previous_update[idx][1] + (
                1 - self.exponential_decay
            ) * (
                update_B ** 2
            )

            # Update weights
            layer.weights = (1 - self.weight_decay) * (layer.weights - update_W)

            # Update biases
            layer.biases -= update_B
