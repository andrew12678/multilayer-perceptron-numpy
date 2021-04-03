from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear
from collections import namedtuple

Parameter = namedtuple("Parameter", ["weights", "biases"])


class Momentum(Optimiser):
    def __init__(
        self,
        layers: List[Linear],
        learning_rate: float,
        weight_decay: float = 0.01,
        momentum: float = 0.9,
    ):
        """
        Updates the parameters at the end of each batch in SGD with momentum fashion.
        Args:
            layers (List[Linear]): List of parameters holding layers (making the assumption that we are working with
                                    linear layers but if this were more general we would make a "Parameters" class that
                                    could be updated unique for each instance.
            learning_rate (float): the learning rate of the optimiser.
            momentum (float): the momentum parameter.
            weight_decay (float): the decay factor applied to the weights before an update.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = [Parameter(0, 0) for _ in range(len(layers))]
        self.weight_decay = weight_decay

    def step(self):
        for idx, layer in enumerate(self.layers):
            new_weight_velocity = (
                self.velocity[idx].weights * self.momentum + self.lr * layer.grad_W
            )
            new_bias_velocity = (
                self.velocity[idx].biases * self.momentum + self.lr * layer.grad_b
            )
            self.velocity[idx] = Parameter(new_weight_velocity, new_bias_velocity)
            layer.weights = (1 - self.weight_decay) * (
                layer.weights - new_weight_velocity
            )
            layer.biases -= new_bias_velocity
