import numpy as np
from .layer import Layer
from ..utils.ml import initialise_weights


class Linear(Layer):
    def __init__(
        self, n_in: int, n_out: int, activation_fn: str, dropout_rate: float = 0
    ):

        """
        Sets up a linear layer taking in n_in inputs and outputs n_out outputs
        Args:
            n_in (int): number of inputs to take for each neuron in the layer
            n_out (int): number of outputs/neurons
            activation_fn (str): type of activation function used for weight initialisation (he, xavier, xavier4)
            dropout_rate (float): the probability/ratio of neurons to be dropped out for the layer (0<=p<=1)
        """

        # Make superclass
        super().__init__()

        # Initialise the weights of the layer
        self.weights = initialise_weights(n_in, n_out, activation_fn)

        # A bias for each output/neuron
        self.biases = np.zeros(n_out)

        # Set gradients as the size of respective arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)

        # Set dropout rate as attribute
        self.dropout_rate = dropout_rate

        # Create dropout vector that contains which neurons to switch off
        self.dropout_mask = np.random.binomial(size=n_in, n=1, p=1 - dropout_rate)

        # Initialise input & output attributes to store later
        self.input = None
        self.output = None

        # Initialise layer in training mode
        self.training = True

    # Complete feed-forward pass for current layer
    def forward(self, x: np.ndarray):

        """
        Computes the forward pass of the layer
        Args:
            x (np.ndarray): the input array for current batch (shape: batch size, features)
        Returns:
            output (np.ndarray): net output of the linear transformation (pre-activation)
        """

        # Check if currently training
        if self.training:
            # Perform dropout/scaling and set input
            self.input = (x * self.dropout_mask) * (1 / (1 - self.dropout_rate))

        # Check if being used for testing/inference
        else:
            # Use standard network with no dropout/scaling
            self.input = x

        # Calculate the net output of the current layer
        net_output = np.dot(self.input, self.weights) + self.biases

        # Set output of current layer
        self.output = net_output

        # Return output
        return self.output

    # Complete backward pass for current layer
    def backward(self, upstream_grad: np.ndarray):

        """
        Computes the backward pass of the layer
        Args:
            upstream_grad (np.ndarray): the gradient from the upstream

        Returns:

        """

        # Set weight gradients using the backward-passed sensitivity from following layer
        self.grad_W = np.dot(self.input.T, upstream_grad)

        # Calculate bias gradients
        self.grad_b = upstream_grad.sum(axis=0)

        # Calculate sensitivity/delta array
        delta = np.dot(upstream_grad, self.weights.T)

        # Return sensitivity/delta for backwards propagation of the error
        return delta

    # Initialise empty gradient arrays for weights and bias
    def zero_grad(self):

        # Initialise empty gradient arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)
