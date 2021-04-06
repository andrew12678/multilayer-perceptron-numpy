from .layer import Layer
import numpy as np


class Batch_Norm(Layer):
    def __init__(self,
                 n_in: int,
                 learning_rate: float,
                 stability_constant: float = 1e-9):

        """
        Sets up a batch normalisation layer taking in n_in inputs and outputs n_in outputs
        Args:
            n_in (int): number of input neurons
        """

        # Make superclass
        super().__init__()

        # Initialise input & output attributes to store later
        self.input = None
        self.output = None

        # Intialise learning rate attribute
        self.learning_rate = learning_rate

        # Initialise stability constant to prevent dividing by zero
        self.epsilon = stability_constant

        # Initialise scale and shift parameters
        self.gamma = np.ones(n_in,)
        self.beta = np.zeros(n_in,)

        # Initialise batch normalisation metrics
        self.batch_means = None
        self.batch_variances = None
        self.batch_norm = None
        self.batch_size = None

        # Complete feed-forward pass for current layer
        def forward(self, X_array: np.ndarray):

            """
            Computes the forward pass of the layer
            Args:
                X (np.ndarray): the input array

            Returns:

            """

            # Store raw input for current batch-norm layer
            self.input = X_array
            self.batch_size = np.shape(self.input)[0]

            # Get means for each feature/column
            self.batch_means = np.mean(self.input, axis=0)

            # Get standard deviations for each feature/column
            self.batch_variances = np.std(self.input, axis=0)

            # Normalise batch array column-wise
            self.batch_norm = (self.input - self.batch_means) / np.sqrt(self.batch_variances + self.epsilon)

            # Perform linear transform and store as output
            self.output = (self.gamma * self.batch_norm) + self.beta

            # Return normalised batch array
            return self.output

        # Complete backward pass for current batch-norm layer
        def backward(self, upstream_grad: np.ndarray):

            """
            Computes the backward pass of the layer
            Args:
                upstream_grad (np.ndarray): the gradient from the upstream for each batch sample (rows) and output (cols)

            Returns:

            """

            # Calculate the loss/norm gradient for the entire layer (shape: M, n_out)
            dLoss_dNorm = upstream_grad * self.gamma

            # Calculate the loss/variance gradient for the current layer (shape: 1, n_in)
            a = dLoss_dNorm * (self.input - self.batch_means)
            b = -((self.batch_variances + self.epsilon)**(-3/2))/2
            dLoss_dVariance = np.sum(a * b, axis=0)

            # Calculate the loss/mean gradient for the current layer (shape: 1, n_in)
            a = np.sum(dLoss_dNorm * (-1/np.sqrt(self.batch_variances + self.epsilon)), axis=0)
            b = dLoss_dVariance * (np.sum(-2 * (self.input - self.batch_means), axis=0))/self.batch_size
            dLoss_dMean = a + b

            # Calculate the loss/input gradient for the current layer (shape: M, n_in)
            a = dLoss_dNorm * (1/np.sqrt(self.batch_variances + self.epsilon))
            b = dLoss_dVariance * (2 * (self.input - self.batch_means)/self.batch_size)
            c = dLoss_dMean * (1 / self.batch_size)
            dLoss_dInput = a + b + c

            # Calculate the loss/Gamma gradient for the current layer (shape: 1, n_in)
            dLoss_dGamma = np.sum(upstream_grad * self.batch_norm, axis=0)

            # Calculate the loss/Beta gradient for the current layer (shape: 1, n_in)
            dLoss_dBeta = np.sum(upstream_grad, axis=0)

            # Update gamma/scale parameter
            self.gamma -= (self.learning_rate * dLoss_dGamma * self.batch_norm).sum(axis=0)

            # Calculate bias gradients
            self.beta += dLoss_dBeta

            # Calculate sensitivity/delta array
            delta = dLoss_dInput

            # Return sensitivity/delta and dropout
            return delta