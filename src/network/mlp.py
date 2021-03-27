import numpy as np
from typing import List, Tuple
from ..layers.linear import Linear
from ..utils.helpers import create_activation_layer


class MLP:
    def __init__(self, layer_sizes: List[Tuple], activation: List[str]):
        """
        Initialises the layers of the Multilayer Perceptron
        Args:
            layer_sizes (List[Tuple]): layers of the network as tuples (in, out)
            activation (List[str): activation for each layer
            loss (str): the loss used as the criterion of the network
        """
        self.layers = []

        for idx, (layer_size, act) in enumerate(zip(layer_sizes, activation)):
            input_size, output_size = layer_size
            if idx and layer_sizes[idx - 1][1] != input_size:
                raise ValueError(
                    "Each layer's output size must be next layer's input size"
                )
            self.layers.append(Linear(input_size, output_size))
            self.layers.append(create_activation_layer(act))

    def forward(self, input: np.ndarray):
        """
        Propagates input through each layer
        Args:
            input (np.ndarray): the input array

        Returns:

        """
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    def backward(self, upstream_grad):
        """
        Back propagates gradients
        Returns:

        """
        delta = upstream_grad
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
