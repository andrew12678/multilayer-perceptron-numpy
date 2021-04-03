from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP

import numpy as np


def run():
    # Set a random seed to reproduce results
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_directory("data")
    n_clasess = len(np.unique(y_train))
    layer_sizes = [(X_train.shape[1], 128), (128, 64), (64, n_clasess)]
    activations = ["relu", "relu", None]

    model = MLP(layer_sizes, activations)
    trainer = Trainer(
        X_train,
        y_train,
        model,
        64,
        100,
        "cross_entropy",
        "sgd",
        learning_rate=0.01,
        weight_decay=0.0,
        momentum=0.9,
    )
    trainer.train()


if __name__ == "__main__":
    run()
