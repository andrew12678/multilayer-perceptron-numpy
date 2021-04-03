from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear


class SGD(Optimiser):
    def __init__(self, layers: List[Linear], learning_rate: float, weight_decay: float):
        """
        Updates the parameters at the end of each batch in SGD fashion
        Args:
            layers (List[Linear]): List of parameters holding layers (making the assumption that we are working with
                                    linear layers but if this were more general we would make a "Parameters" class that
                                    could be updated unique for each instance
            learning_rate (float): the learning rate of the optimiser
            weight_decay (float): the decay factor applied to the weights before an update.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.layers:
            layer.weights = (1 - self.weight_decay) * (layer.weights - layer.grad_W)
            layer.biases -= layer.grad_b
