import numpy as np
from ..network.mlp import MLP
from ..layers.layer import Layer
from ..utils.ml import Batcher, one_hot
from ..utils.helpers import create_loss_function, create_optimiser


class Trainer:
    """
    Class to initialise network architecture, loss function, optimiser and run training
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
        self.X = X
        self.y = one_hot(y)
        self.model = model
        # Create batcher object to handle mini-batch creation
        self.batcher = Batcher(self.X.shape[0], batch_size)
        self.n_epochs = n_epochs
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

        # Loop through designated number of epochs
        for epoch in range(1, self.n_epochs + 1):

            # Display epoch numbers
            print(f"Starting epoch: {epoch}")

            # Zeroing the gradients
            self.model.zero_grad()

            # Get batches indices for this epoch
            batches = self.batcher.generate_batch_indices()
            acc_loss = 0
            for batch in batches:
                # Get current batch
                X_batch, y_batch = self.X[batch], self.y[batch]

                # Get output from network for batch
                output = self.model.forward(X_batch)

                # Calculate loss for current batch
                loss = self.loss(y_batch, output)

                # Display loss
                print(f"Loss {loss}")

                # Perform backward propagation and get sensitivity for output layer
                delta = self.loss.backward()

                # Perform backward propagation for all layers
                self.model.backward(delta)

                # Update weights
                self.optimiser.step()

                acc_loss += loss
            if epoch % 5 == 4:
                print(f"Epoch: {epoch + 1}, loss: {acc_loss / len(batches)}")
