from .layer import Layer
import numpy as np


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

        # Check if Xavier initialisation (Sigmoid/Logistic)
        if activation_fn == "logistic":

            # Randomly initialise weights according to Xavier
            self.weights = np.random.uniform(
                low=-np.sqrt(6.0 / (n_in + n_out)),
                high=np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out),
            )

        # Check if Xavier initialisation (Tanh)
        elif activation_fn == "tanh":

            # Initialise Xavier weights and multiply by 4
            self.weights = (
                np.random.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high=np.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out),
                )
                * 4
            )

        # Check if He initialisation (ReLU & LReLU)
        elif activation_fn == "relu" or activation_fn == "leaky_relu":

            # Initialise weights using Kaiming uniform distribution
            self.weights = np.random.uniform(
                low=-np.sqrt(6.0 / n_in),
                high=np.sqrt(6.0 / n_in),
                size=(n_in, n_out),
            )

        # Check if activation is not defined (REVIEW THIS)
        elif activation_fn == None:

            # Initialise weights using Kaiming uniform distribution
            self.weights = np.random.uniform(
                low=-np.sqrt(6.0 / n_in),
                high=np.sqrt(6.0 / n_in),
                size=(n_in, n_out),
            )

        # Check if unknown activation function entered
        else:

            # Raise exception indicating that initialisation method is unknown
            raise ValueError(
                "Activation function not recognised for weight initialisation."
            )

        # A bias for each output/neuron
        self.biases = np.zeros(
            n_out,
        )

        # Set gradients as the size of respective arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)

        # Set dropout ratio as attribute
        self.dropout_rate = dropout_rate

        # Create dropout vector that contains which neurons to switch off
        self.dropout_array = np.random.binomial(size=n_in, n=1, p=1 - dropout_rate)

        # Initialise input & output attributes to store later
        self.input = None
        self.output = None

    # Complete feed-forward pass for current layer
    def forward(self, X: np.ndarray):

        """
        Computes the forward pass of the layer
        Args:
            X (np.ndarray): the input array

        Returns:

        """

        # Perform dropout/scaling and set input
        self.input = (X * self.dropout_array) * (1 / (1 - self.dropout_rate))

        # Calculate the net output of the current layer
        net_output = np.dot(self.input, self.weights) + self.biases

        # Apply dropout and set output of current layer
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

        # Return sensitivity/delta and dropout
        return delta

    # Initialise empty gradient arrays for weights and bias
    def zero_grad(self):

        # Initialise empty gradient arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)
