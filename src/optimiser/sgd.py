from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear


class SGD(Optimiser):
    def __init__(self, layers: List[Linear], learning_rate: float):
        """
        Updates the parameters at the end of each batch in SGD fashion
        Args:
            layers (List[Linear]): List of parameters holding layers (making the assumption that we are working with
                                    linear layers but if this were more general we would make a "Parameters" class that
                                    could be updated unique for each instance
            learning_rate (float): the learning rate of the optimiser
        """
        super().__init__(layers)
        self.lr = learning_rate

    def step(self):

        # Loop through all layers of MLP
        for layer in self.layers:

            # Update weights and biases
            layer.weights -= layer.grad_W
            layer.biases -= layer.grad_B
