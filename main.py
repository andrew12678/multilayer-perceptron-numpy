from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import one_hot, create_stratified_kfolds
import numpy as np


def run():

    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Define the number of classes
    n_classes = len(np.unique(y_train))

    # Define layers in network (input_dim, output_dim)
    layer_sizes = [(X_train.shape[1], 128), (128, 64), (64, n_classes)]

    # Define activation functions for each layer
    activations = ["relu", "relu", None]

    # Define dropout rates for all layers
    dropout_rates = [0, 0.5, 0.5]

    # Create multi-layer perceptron model (i.e build model object)
    model = MLP(
        layer_sizes=layer_sizes,
        activations=activations,
        dropout_rates=dropout_rates,
        batch_normalisation=True,
    )

    # Create trainer object to handle train each epoch (define input data and parameters)
    trainer = Trainer(
        X=X_train,
        y=y_train,
        model=model,
        batch_size=64,
        n_epochs=3,
        loss="cross_entropy",
        optimiser="adadelta",
        learning_rate=0.001,
        weight_decay=0.0,
        momentum=0.9,
    )

    # Train model
    trained_model = trainer.train()

    # Test model
    metrics = trainer.validation(X=X_test, y=y_test)
    print(metrics)


def run_kfolds():

    # Load training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
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
    batch_normalisation = True

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    # Instantiate accumulated loss tracker
    acc_loss = 0

    # Train and test on each k-fold split
    for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):

        # Define model using current architecture
        model = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            batch_normalisation=batch_normalisation,
        )

        # Train model using current hyperparameters
        trainer = Trainer(
            X=X_train_k,
            y=y_train_k,
            model=model,
            batch_size=batch_size,
            n_epochs=n_epochs,
            loss=loss_fn,
            optimiser=optimiser,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        # Train model on training set
        trainer.train()

        # Perform validation
        # Extract validation loss and predicted labels
        results = trainer.validation(X=X_val_k, y=y_val_k)

        # Get model predictions for validation set
        # fold_preds = model.forward(X_val_k)

        # Get model loss on validation set
        # fold_loss = trainer.loss(one_hot(y_val_k), fold_preds)

        # Display loss for current fold
        print(f"Loss for fold {k + 1}: {results['loss']}")
        print(f"accuracy: {results['accuracy']}")
        print(f"f1_macro: {results['f1_macro']}")

        # Add loss to total
        acc_loss += results["loss"]

    # Display the overall cross-validation loss across all folds
    print(f"Overall cross-validation loss: {acc_loss}")


# Run script
if __name__ == "__main__":
    # Set a random seed to reproduce results
    np.random.seed(42)
    run()
    # run_kfolds()
