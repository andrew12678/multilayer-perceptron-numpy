import numpy as np
import time
import yaml
import itertools

from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import one_hot, create_stratified_kfolds, create_layer_sizes


def run():

    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Define the number of classes
    n_classes = len(np.unique(y_train))

    # Load hyperparameters from file
    with open("hyperparams/config.yml", "r") as f:
        p = yaml.safe_load(f)["params1"]

    # Print hyperparameters
    for key, value in p.items():
        print(key, ":", value)

    # Create a list of tuples indicating the size of each network layer
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, p["num_hidden"], p["hidden_size"]
    )

    # Create multi-layer perceptron model (i.e build model object)
    model = MLP(
        layer_sizes=layer_sizes,
        activations=p["activations"],
        dropout_rates=p["dropout_rates"],
        batch_normalisation=p["batch_normalisation"],
    )

    # Create trainer object to handle train each epoch (define input data and parameters)
    trainer = Trainer(
        X=X_train,
        y=y_train,
        model=model,
        batch_size=p["batch_size"],
        n_epochs=p["num_epochs"],
        loss=p["loss"],
        optimiser=p["optimiser"],
        learning_rate=p["learning_rate"],
        weight_decay=p["weight_decay"],
        momentum=p["momentum"],
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

    # Load hyperparameters from file
    with open("hyperparams/config.yml", "r") as f:
        hyperparam_grid = yaml.safe_load(f)["grid2"]

    # Print hyperparameters
    for key, value in hyperparam_grid.items():
        print(key, ":", value)

    # Setup grid search by enumerating all possible combination of parameters in hyperparam_grid
    # e.g. {'batch_size': [24, 48], 'num_epochs': [2]} -> [{'batch_size': 24, 'num_epochs': 2}
    #                                                     {'batch_size': 48, 'num_epochs': 2}]
    keys, values = zip(*hyperparam_grid.items())
    param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    for p in param_list:
        # Create a list of tuples indicating the size of each network layer
        layer_sizes = create_layer_sizes(
            X_train.shape[1], n_classes, p["num_hidden"], p["hidden_size"]
        )

        # Instantiate accumulated loss tracker
        acc_loss = 0

        # Train and test on each k-fold split
        for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):

            # Define model using current architecture
            model = MLP(
                layer_sizes=layer_sizes,
                activations=p["activations"],
                dropout_rates=p["dropout_rates"],
                batch_normalisation=p["batch_normalisation"],
            )

            # Create trainer object to handle train each epoch (define input data and parameters)
            trainer = Trainer(
                X=X_train,
                y=y_train,
                model=model,
                batch_size=p["batch_size"],
                n_epochs=p["num_epochs"],
                loss=p["loss"],
                optimiser=p["optimiser"],
                learning_rate=p["learning_rate"],
                weight_decay=p["weight_decay"],
                momentum=p["momentum"],
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
            # print(f"Loss for fold {k + 1}: {results['loss']}")
            # print(f"accuracy: {results['accuracy']}")
            # print(f"f1_macro: {results['f1_macro']}")

            # Add loss to total
            acc_loss += results["loss"]

        print(f"Trained on params:")
        for k, v in p.items():
            print(f"{k}: {v}")
        # Display the overall cross-validation loss across all folds
        print(f"Overall cross-validation loss: {acc_loss}")


# Run script
if __name__ == "__main__":
    start_time = time.time()
    # Set a random seed to reproduce results
    np.random.seed(42)
    # run()
    run_kfolds()

    print("Run time: ", time.time() - start_time)
