from .optimiser import Optimiser
from typing import List
from ..layers.linear import Linear
from ..layers.batch_norm import BatchNorm
from collections import namedtuple

Parameter = namedtuple("Parameter", ["weights", "biases"])
Parameter_BN = namedtuple("Parameter", ["gamma", "beta"])


class Momentum(Optimiser):
    def __init__(
        self, layers: List, learning_rate: float, momentum: float = 0.9
    ):
        """
        Updates the parameters at the end of each batch in SGD with momentum fashion
        Args:
            layers (List[Linear]): List of parameters holding layers (making the assumption that we are working with
                                    linear layers but if this were more general we would make a "Parameters" class that
                                    could be updated unique for each instance
            learning_rate (float): the learning rate of the optimiser
            momentum (float): the momentum parameter
        """

        # Superclass
        super().__init__(layers)

        # Set learning rate and momentum attributes
        self.lr = learning_rate
        self.momentum = momentum

        # Create lists to store velocities of each layer's learnable parameters
        self.velocity = [Parameter(0, 0) for _ in range(len(layers))]

    # Step function that optimises all layers
    def step(self):

        # Loop through all layers
        for idx, layer in enumerate(self.layers):

            # Update velocity for weights
            new_weight_velocity = (
                self.velocity[idx].weights * self.momentum + self.lr * layer.grad_W
            )

            # Update velocity for biases
            new_bias_velocity = (
                self.velocity[idx].biases * self.momentum + self.lr * layer.grad_b
            )

            # Store velocities
            self.velocity[idx] = Parameter(new_weight_velocity, new_bias_velocity)

            # Update weights & biases
            layer.weights -= new_weight_velocity
            layer.biases -= new_bias_velocity

