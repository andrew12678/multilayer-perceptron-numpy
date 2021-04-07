import numpy as np
from typing import List, Tuple
from ..layers.linear import Linear
from ..utils.helpers import create_activation_layer


class MLP:
    def __init__(
        self,
        layer_sizes: List[Tuple],
        activation: List[str],
        dropout_rates: List[float],
    ):

        """
        Initialises the layers of the Multilayer Perceptron
        Args:
            layer_sizes (List[Tuple]): layers of the network as tuples (in, out)
            activation (List[str): activation for each layer
            loss (str): the loss used as the criterion of the network
            dropout_rates (List[float]): the dropout rates for each layer
        """

        # Initialise empty list to append hidden layer objects to
        self.layers = []

        # Iterate over all layers and respective activation functions
        for idx, (layer_size, act, dropout_rate) in enumerate(
            zip(layer_sizes, activation, dropout_rates)
        ):

            # Extract input and output dimensions
            input_size, output_size = layer_size

            # Check if output dimension of previous layer is equal to input of current layer
            if idx and layer_sizes[idx - 1][1] != input_size:

                # Raise error
                raise ValueError(
                    "Each layer's output size must be next layer's input size"
                )

            # Define current hidden layer and append object to list
            self.layers.append(
                Linear(
                    n_in=input_size,
                    n_out=output_size,
                    activation_fn=act,
                    dropout_rate=dropout_rate,
                )
            )

            # Check that activation function type is defined
            if act:

                # Append activation layer to layers list
                self.layers.append(create_activation_layer(act))

    def forward(self, input: np.ndarray):

        """
        Propagates input through each layer
        Args:
            input (np.ndarray): the input array

        Returns:

        """

        # Iterate through all layer objects
        for layer in self.layers:

            # Calculate output from forward pass on current layer
            output = layer.forward(input)

            # Set input for next layer as current output
            input = output

        # Return output array of current layer
        return output

    # Back-propogation for all layers
    def backward(self, upstream_grad):

        """
        Back propagates gradients
        Returns:

        """

        # Set sensitivity as upstream gradient
        delta = upstream_grad

        # Loop through all layers in reverse
        for layer in reversed(self.layers):

            # Perform backward pass on current layer, extracting current propagated sensitivity
            delta = layer.backward(delta)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
