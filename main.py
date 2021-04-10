from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import one_hot, create_stratified_kfolds
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

    # Create multi-layer perceptron model (i.e build model object)
    model = MLP(
        layer_sizes=layer_sizes,
        activation=activations,
        dropout_rates=dropout_rates,
        batch_normalisation=True,
    )

    # Create trainer object to handle train each epoch (define input data and parameters)
    trainer = Trainer(
        X=X_train,
        y=y_train,
        model=model,
        batch_size=64,
        n_epochs=10,
        loss="cross_entropy",
        optimiser="sgd",
        learning_rate=0.001,
        weight_decay=0.0,
        momentum=0.9,
    )

    # Train model
    trainer.train()


def run_kfolds():
    X_train, y_train, X_test, y_test = load_directory("data")
    n_classes = len(np.unique(y_train))

    num_folds = 5
    # At some point we will do a search over some of these, and justify the values of others theoretically.
    batch_size = 64
    n_epochs = 5
    loss_fn = "cross_entropy"
    optimiser = "sgd"
    learning_rate = 0.01
    weight_decay = 0.0
    momentum = 0.9
    layer_sizes = [(X_train.shape[1], 128), (128, 64), (64, n_classes)]
    activations = ["relu", "relu", None]
    dropout_rates = [0, 0.5, 0.5]

    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    loss = 0
    # Train and test on each k-fold split
    for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):
        model = MLP(layer_sizes, activations, dropout_rates)
        trainer = Trainer(
            X_train_k,
            y_train_k,
            model,
            batch_size,
            n_epochs,
            loss_fn,
            optimiser,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        # Train model on training set
        trainer.train()
        # Get model predictions for validation set
        fold_preds = model.forward(X_val_k)
        # Get model loss on validation set
        fold_loss = trainer.loss(one_hot(y_val_k), fold_preds)
        print(f"Loss for fold {k}: {fold_loss}")
        loss += fold_loss
    print(f"Overall cross-validation loss: {loss}")


# Run script
if __name__ == "__main__":
    # Set a random seed to reproduce results
    np.random.seed(42)
    run()
    # run_kfolds()
