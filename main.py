from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP

import numpy as np


def run():

    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Define the number of classes
    n_clasess = len(np.unique(y_train))

    # Define layers in network (input_dim, output_dim)
    layer_sizes = [(X_train.shape[1], 128), (128, 64), (64, n_clasess)]

    # Define activation functions for each layer
    activations = ["relu", "relu", None]

    # Define dropout rates for all layers
    dropout_rates = [0, 0.5, 0.5]

    # Build multi-layer perceptron model (i.e build model object)
    model = MLP(layer_sizes=layer_sizes,
                activation=activations,
                dropout_rates=dropout_rates,
                batch_normalisation=True)

    # Builder trainer object (define input data and parameters)
    trainer = Trainer(X=X_train,
                      y=y_train,
                      model=model,
                      batch_size=64,
                      n_epochs=10,
                      loss="cross_entropy",
                      optimiser="momentum",
                      learning_rate=0.001)

    # Train MLP model
    trainer.train()

# Run script
if __name__ == "__main__":
    run()
