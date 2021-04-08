from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear
from ..layers.batch_norm import BatchNorm



class SGD(Optimiser):
    def __init__(self, layers: List, learning_rate: float):
        """
        Updates the parameters at the end of each batch in SGD fashion
        Args:
            layers (List): List of parameters holding layers (making the assumption that we are working with
                           linear layers but if this were more general we would make a "Parameters" class that
                           could be updated unique for each instance
            learning_rate (float): the learning rate of the optimiser
        """
        super().__init__(layers)
        self.lr = learning_rate

    # Step function that optimises all layers
    def step(self):

        # Loop through all layers of MLP
        for layer in self.layers:

            # Check if layer is a batch normalisation layer
            if isinstance(layer, BatchNorm):

                # Update scale and shift parameters
                layer.gamma -= self.lr * layer.grad_gamma
                layer.beta -= self.lr * layer.grad_beta

            # If layer is any other type (e.g. linear, activation, etc.)
            else:

                # Update weights and biases
                layer.weights -= self.lr * layer.grad_W
                layer.biases -= self.lr * layer.grad_b


