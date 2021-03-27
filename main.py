from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP

import numpy as np


def run():
    X_train, y_train, X_test, y_test = load_directory("data")
    n_clasess = len(np.unique(y_train))
    layer_sizes = [(X_train.shape[1], 128), (128, 64), (64, n_clasess)]
    activations = ["relu", "relu", "relu"]

    model = MLP(layer_sizes, activations)
    trainer = Trainer(
        X_train, y_train, model, 64, 100, "cross_entropy", "momentum", 0.001
    )
    trainer.train()


if __name__ == "__main__":
    run()
