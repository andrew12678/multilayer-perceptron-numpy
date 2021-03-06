from .layer import Layer
import numpy as np


class BatchNorm(Layer):
    def __init__(self, n_in: int, momentum_ma: float = 0.9):

        """
        Sets up a batch normalisation layer taking in n_in inputs and producing the same no. of outputs
        Args:
            n_in (int): number of input neurons and output neurons
            momentum_ma (float): the BatchNorm momentum for the running mean and variances, don't confuse with the SGD
                              momentum
        """

        super().__init__()

        # Initialise input & output attributes to store later
        self.input = None
        self.output = None

        # Store momentum for moving average during inference
        self.momentum = momentum_ma

        # Initialise stability constant to prevent dividing by zero
        self.epsilon = 1e-9

        # Initialise scale (gamma) and shift (beta) parameters
        # NOTE: weights = gamma & biases = beta for continuity in updates via optimiser
        self.weights = np.ones(n_in)
        self.biases = np.zeros(n_in)

        # Set gradients as the size of respective arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)

        # Initialise batch normalisation metrics
        self.batch_means = None
        self.batch_variances = None
        self.batch_norm = None
        self.batch_size = None

        self.running_mean = np.zeros(n_in)
        self.running_var = np.zeros(n_in)

        # Set mode (train/test)
        self.training = True

    # Complete feed-forward pass for current layer
    def forward(self, x: np.ndarray):

        """
        Computes the forward pass of the layer
        Args:
            x (np.ndarray): the input array
        Returns:
            output array
        """

        # Check if model is being trained
        if self.training:

            # Store raw input for current batch-norm layer
            self.input = x
            self.batch_size = self.input.shape[0]

            # Get means for each feature/column
            self.batch_means = self.input.mean(axis=0)

            # Get variances for each feature/column, np.var does population variance so we aren't at risk
            self.batch_variances = self.input.var(axis=0)

            # Normalise batch array column-wise
            self.batch_norm = (self.input - self.batch_means) / np.sqrt(
                self.batch_variances + self.epsilon
            )

            # Perform linear transform and store as output
            self.output = (self.weights * self.batch_norm) + self.biases

            # Saving the running mean and variance with BN momentum
            self.running_mean = (
                self.momentum * self.running_mean
                + (1.0 - self.momentum) * self.batch_means
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1.0 - self.momentum) * self.batch_variances
            )

            # Return normalised batch array
            return self.output

        # Check if model is being tested
        else:

            # Normalise using the running mean and variances
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

            # Using gamma and beta and don't save this variable
            output = self.weights * x_hat + self.biases

            # Return normalised output for inference
            return output

    # Complete backward pass for current batch-norm layer
    def backward(self, upstream_grad: np.ndarray):

        """
        Computes the backward pass of the layer
        Args:
            upstream_grad (np.ndarray): the gradient from the upstream for each batch sample (rows) and output (cols)
        Returns:

        """

        # Calculate the loss/norm gradient for the entire layer (shape: M, n_out)
        dLoss_dNorm = upstream_grad * self.weights

        # Calculate the loss/variance gradient for the current layer (shape: 1, n_in)
        a = dLoss_dNorm * (self.input - self.batch_means)
        b = -((self.batch_variances + self.epsilon) ** (-3 / 2)) / 2
        dLoss_dVariance = np.sum(a * b, axis=0)

        # Calculate the loss/mean gradient for the current layer (shape: 1, n_in)
        a = np.sum(
            dLoss_dNorm * (-1 / np.sqrt(self.batch_variances + self.epsilon)), axis=0
        )
        b = (
            dLoss_dVariance
            * (np.sum(-2 * (self.input - self.batch_means), axis=0))
            / self.batch_size
        )
        dLoss_dMean = a + b

        # Calculate the loss/input gradient for the current layer (shape: M, n_in)
        a = dLoss_dNorm * (1 / np.sqrt(self.batch_variances + self.epsilon))
        b = dLoss_dVariance * (2 * (self.input - self.batch_means) / self.batch_size)
        c = dLoss_dMean * (1 / self.batch_size)
        dLoss_dInput = a + b + c

        # Calculate the loss/Gamma gradient for the current layer (shape: 1, n_in)
        dLoss_dGamma = np.sum(upstream_grad * self.batch_norm, axis=0)

        # Calculate the loss/Beta gradient for the current layer (shape: 1, n_in)
        dLoss_dBeta = np.sum(upstream_grad, axis=0)

        # Update gamma/scale parameter
        self.grad_W = dLoss_dGamma

        # Update gamma/scale parameter
        self.grad_b = dLoss_dBeta

        # Calculate sensitivity/delta array
        delta = dLoss_dInput

        # Return sensitivity/delta
        return delta

    # Initialise empty gradient arrays for weights and bias
    def zero_grad(self):

        # Initialise empty gradient arrays
        self.grad_W = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)
