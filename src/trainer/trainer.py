import numpy as np
from ..network.mlp import MLP
from ..layers.layer import Layer
from ..utils.ml import Batcher, one_hot
from ..utils.helpers import create_loss_function, create_optimiser
from ..utils.metrics import calculate_metrics


class Trainer:

    """
    Class to initialise network architecture, loss function, optimiser and run training
    Also has method for testing and validation
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: MLP,
        batch_size: int,
        n_epochs: int,
        loss: str,
        optimiser: str,
        learning_rate: float,
        weight_decay: float = 0,
        momentum: float = 0,
        exponential_decay: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):

        # Set attributes
        self.X = X
        self.y = one_hot(y)
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Create batcher object to handle mini-batch creation
        self.batcher = Batcher(self.X.shape[0], self.batch_size)

        # Set loss function
        self.loss = create_loss_function(loss)

        # Create optimiser
        self.optimiser = create_optimiser(
            optimiser,
            [l for l in model.layers if isinstance(l, Layer)],
            learning_rate,
            weight_decay,
            momentum,
            exponential_decay,
            beta1,
            beta2,
        )

    # Primary method that trains the network
    def train(self):

        # Ensure model is in training mode
        self.model.train()

        # Instantiate accumulated loss variable for training
        acc_loss = 0

        # Loop through designated number of epochs
        for epoch in range(1, self.n_epochs + 1):

            # Display epoch numbers
            # print(f"Starting epoch: {epoch}")

            # Zeroing the gradients
            self.model.zero_grad()

            # Get batches indices for this epoch
            batches = self.batcher.generate_batch_indices()

            # Loop through all batches
            for batch in batches:

                # Get current batch data
                X_batch, y_batch = self.X[batch], self.y[batch]

                # Get output from network for batch
                output = self.model.forward(X_batch)

                # Calculate loss for current batch
                loss = self.loss(y_batch, output)

                # Perform backward propagation and get sensitivity for output layer
                delta = self.loss.backward()

                # Perform backward propagation for all layers
                self.model.backward(delta)

                # Update weights
                self.optimiser.step()

                # Add batch loss to accumulated loss
                acc_loss += loss

            # Check if epoch is multiple of 5
            if epoch % 5 == 0:
                # Display average loss
                print(f"Epoch: {epoch}, loss: {acc_loss / (epoch * len(batches))}")
                # Kill the training if our acc_loss is a nan
                if np.isnan(acc_loss):
                    return np.nan

        # Return training loss
        return acc_loss / (self.n_epochs * len(batches))

    # Secondary method that evaluates the network performance on a separate test dataset
    def validation(self, X: np.ndarray, y: np.ndarray):

        # Ensure model mode is set to testing
        self.model.test()

        # One-hot testing labels
        y = one_hot(y)

        # Get batches indices for all test data
        batches = Batcher(X.shape[0], self.batch_size).generate_batch_indices()

        # Instantiate accumulated loss variable
        acc_loss = 0

        # Create empty lists for storing predicted and true values in the same order as batches
        y_hat, y_true = [], []

        # Loop through all test batches
        for batch in batches:

            # Get current batch data
            X_batch, y_batch = X[batch], y[batch]

            # Forward pass all data and get output
            batch_output = self.model.forward(X_batch)

            # Calculate average loss for current test batch
            batch_loss = self.loss(y_batch, batch_output)

            # Add batch loss to accumulated loss
            acc_loss += batch_loss

            # Append predicted classes for current test batch to test dataset list
            y_hat.append(np.argmax(batch_output, axis=1))
            y_true.append(np.argmax(y_batch, axis=1))

        # Calculate average loss across all test batches
        test_loss = acc_loss / len(batches)

        # Flatten predicted and true labels in 1d arrays
        y_hat, y_true = np.concatenate(y_hat).ravel(), np.concatenate(y_true).ravel()

        # Calculate metrics
        metrics_dict = calculate_metrics(y=y_true, y_hat=y_hat)

        # Add loss to metrics dictionary
        metrics_dict["loss"] = test_loss

        # Return loss and predictive accuracy of model
        return metrics_dict
