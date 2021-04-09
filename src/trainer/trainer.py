import numpy as np
from ..network.mlp import MLP
from ..layers.layer import Layer
from ..utils.ml import Batcher, one_hot
from ..utils.helpers import create_loss_function, create_optimiser


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
        momentum: float = None,
    ):

        # Set attributes
        self.X = X
        self.y = one_hot(y)
        self.model = model
        self.n_epochs = n_epochs

        # Create batcher object to handle mini-batch creation
        self.batcher = Batcher(self.X.shape[0], batch_size)

        # Set loss function
        self.loss = create_loss_function(loss)
        self.optimiser = create_optimiser(
            optimiser,
            [l for l in model.layers if isinstance(l, Layer)],
            learning_rate,
            weight_decay,
            momentum,
        )

    # Primary method that trains the network
    def train(self):

        # Ensure model is in training mode
        self.model.train()

        # Loop through designated number of epochs
        for epoch in range(1, self.n_epochs + 1):

            # Display epoch numbers
            print(f"Starting epoch: {epoch}")

            # Zeroing the gradients
            self.model.zero_grad()

            # Get batches indices for this epoch
            batches = self.batcher.generate_batch_indices()

            # Instantiate accumulated loss variable for current epoch
            acc_loss = 0

            # Loop through all batches
            for batch in batches:

                # Get current batch data
                X_batch, y_batch = self.X[batch], self.y[batch]

                # Get output from network for batch
                output = self.model.forward(X_batch)

                # Calculate loss for current batch
                loss = self.loss(y_batch, output)

                # Display batch loss
                #print(f"Loss {loss}")

                # Perform backward propagation and get sensitivity for output layer
                delta = self.loss.backward()

                # Perform backward propagation for all layers
                self.model.backward(delta)

                # Update weights
                self.optimiser.step()

                # Add batch loss to accumulated loss
                acc_loss += loss

            # Check if epoch is multiple of 5
            if epoch % 5 == 4:

                # Display average loss for all batches in the current epoch
                print(f"Epoch: {epoch + 1}, loss: {acc_loss / len(batches)}")

        # Return trained model
        return self.model

    # Secondary method that evaluates the network performance on a separate test dataset
    def test(self, X: np.ndarray, y: np.ndarray):

        # Ensure model mode is set to testing
        self.model.test()

        # Forward pass all data and get output
        output = self.model.forward(X)

        # One-hot each sample's class
        y = one_hot(y)

        # Calculate average loss for test dataset
        loss = self.loss(y, output)

        # Display test loss
        print(f"Average test loss: {loss}")

        # Return predicted values for model
        return output
