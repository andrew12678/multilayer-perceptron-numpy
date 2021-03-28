import numpy as np
from ..network.mlp import MLP
from ..layers.layer import Layer
from ..utils.ml import Batcher, one_hot
from ..utils.helpers import create_loss_function, create_optimiser


class Trainer:
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
    ):
        self.model = model
        self.X = X
        # Changing y to one_hot
        self.y = one_hot(y)
        self.batcher = Batcher(self.X.shape[0], batch_size)
        self.n_epochs = n_epochs
        self.loss = create_loss_function(loss)
        self.optimiser = create_optimiser(
            optimiser, [l for l in model.layers if isinstance(l, Layer)], learning_rate
        )

    def train(self):
        for i in range(1, self.n_epochs + 1):
            print(f"Starting epoch: {i}")

            # Zeroing the gradients
            self.model.zero_grad()
            # Get batches indices for this epoch
            batches = self.batcher.generate_batch_indices()
            for batch in batches:
                # Get current batch
                X_batch, y_batch = self.X[batch], self.y[batch]
                # Get output from network for batch
                output = self.model.forward(X_batch)

                loss = self.loss(y_batch, output)
                print(f"Loss {loss}")
                delta = self.loss.backward()

                self.model.backward(delta)

                self.optimiser.step()
