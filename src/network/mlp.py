import numpy as np
from typing import List, Tuple
from ..layers.linear import Linear
from ..layers.batch_norm import BatchNorm
from ..utils.helpers import create_activation_layer


class MLP:
    def __init__(
        self,
        layer_sizes: List[Tuple],
        activations: List[str],
        dropout_rates: List[float],
        batch_normalisation: bool = True,
    ):

        """
        Initialises the layers of the Multilayer Perceptron (i.e. builds the network)
        Args:
            layer_sizes (List[Tuple]): layers of the network as tuples (in, out)
            activations (List[str): activation for each layer
            dropout_rates (List[float]): the dropout rates for each layer
            batch_normalisation (bool): indicates whether to use batch normalisation or not (default True)
        """

        # Initialise empty list to append hidden layer objects to
        self.layers = []

        # Initialise model in training mode
        self.training = True

        # Iterate over all layers and respective activation functions
        for idx, (layer_size, act, dropout_rate) in enumerate(
            zip(layer_sizes, activations, dropout_rates)
        ):

            # Extract input and output dimensions
            input_size, output_size = layer_size

            # Check if output dimension of previous layer is equal to input of current layer
            if idx and layer_sizes[idx - 1][1] != input_size:

                # Raise error
                raise ValueError(
                    "Each layer's output size must be next layer's input size."
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

            # Check if batch normalisation is to be used and if before final layer
            if batch_normalisation and idx < len(layer_sizes) - 1:

                # Append batch normalisation layer to layers list
                self.layers.append(BatchNorm(n_in=output_size))

            # Check that activation function type is defined
            if act:

                # Append activation layer to layers list
                self.layers.append(create_activation_layer(act))
    # Complete forward pass of the neural network
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

    # Back-propagation for all layers
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

    # Initialise gradients for gradient descent
    def zero_grad(self):

        # Loop through all layers
        for layer in self.layers:

            # Check if current layer has the zero_grad attribute
            if hasattr(layer, "zero_grad"):

                # Initialise gradients
                layer.zero_grad()

    # Set model mode to training
    def train(self):

        # Set model mode to training
        self.training = True

        # Loop through all layers
        for layer in self.layers:

            # Check if current layer has training attribute
            if hasattr(layer, "training"):

                # Set layer mode to model mode (training)
                layer.training = self.training

    # Set model mode to testing
    def test(self):

        # Set model mode to testing
        self.training = False

        # Loop through all layers
        for layer in self.layers:

            # Check if current layer has training attribute
            if hasattr(layer, "training"):

                # Set layer mode to model mode (testing)
                layer.training = self.training
