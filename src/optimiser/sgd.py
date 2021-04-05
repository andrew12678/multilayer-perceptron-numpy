from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear
from collections import namedtuple

Parameter = namedtuple("Parameter", ["weights", "biases"])


class SGD(Optimiser):
    def __init__(
        self,
        layers: List[Linear],
        learning_rate: float,
        weight_decay: float = 0,
        momentum: float = 0,
    ):
        """
        Updates the parameters at the end of each batch in SGD fashion. Note that when momentum==0, the new_weight_velocity
        term is simply self.lr*layer.grad_W as in vanilla SGD.
        Args:
            layers (List[Linear]): List of parameters holding layers (making the assumption that we are working with
                                    linear layers but if this were more general we would make a "Parameters" class that
                                    could be updated unique for each instance
            learning_rate (float): the learning rate of the optimiser
            weight_decay (float): the decay factor applied to the weights before an update.
            momentum (float): the momentum parameter.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = [Parameter(0, 0) for _ in range(len(layers))]

    def step(self):
        for idx, layer in enumerate(self.layers):
            new_weight_velocity = (
                self.momentum * self.velocity[idx].weights + self.lr * layer.grad_W
            )
            new_bias_velocity = (
                self.momentum * self.velocity[idx].biases + self.lr * layer.grad_b
            )
            self.velocity[idx] = Parameter(new_weight_velocity, new_bias_velocity)
            layer.weights = (1 - self.weight_decay) * (
                layer.weights - new_weight_velocity
            )
            layer.biases -= new_bias_velocity
