import numpy as np
import time
import yaml
import itertools
import argparse
from multiprocessing import Pool
import json

from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import create_stratified_kfolds, create_layer_sizes


# Single model training and testing
def run(args):

    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Define the number of classes
    n_classes = len(np.unique(y_train))

    # Load hyperparameters from file
    with open(args.config, "r") as f:
        p = yaml.safe_load(f)[args.hyperparams]

    # Print hyperparameters
    # for key, value in p.items():
    #     print(key, ":", value)

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
    train_loss = trainer.train()

    # Test model
    test_metrics = trainer.validation(X=X_test, y=y_test)
    print(f"Trained on params: {p}")
    print(f"Overall training loss: {train_loss}")
    print(f"Overall test set results: {test_metrics}")


# Primary function that takes external arguments, performs k-fold cross-validation and returns model results
def run_experiment(args):

    # Extract current set of arguments
    params, X_train, y_train, n_classes, splits, X_test, y_test = args

    # Define layer sizes/network architecture
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, params["num_hidden"], params["hidden_size"]
    )

    # Instantiate accumulated loss tracker
    acc_loss = 0

    # Train and test on each k-fold split
    for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):

        # Define model using current architecture
        model = MLP(
            layer_sizes=layer_sizes,
            activations=params["activations"],
            dropout_rates=params["dropout_rates"],
            batch_normalisation=params["batch_normalisation"],
        )

        # Create trainer object to handle all epochs
        # Define training data, hyperparameters and modules to be used
        trainer = Trainer(
            X=X_train_k,
            y=y_train_k,
            model=model,
            batch_size=params["batch_size"],
            n_epochs=params["num_epochs"],
            loss=params["loss"],
            optimiser=params["optimiser"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"],
        )

        # Train model on training set
        _ = trainer.train()

        # Extract validation loss and predicted labels
        val_results = trainer.validation(X=X_val_k, y=y_val_k)

        # Add loss to total
        acc_loss += val_results["loss"]

    # Get test results for fully-validated model
    test_results = trainer.validation(X=X_test, y=y_test)

    # Display model params and final test results
    print(f"Trained on params: {params}")
    print(f"Overall cross-validation loss: {acc_loss / len(splits)}")
    print(f"Overall test set results: {test_results}")

    # Store parameters and validation/test results in summary dictionary
    summary = {'params': params, 'cv_loss': acc_loss / len(splits), 'test_errors': test_results}

    # Return model summary
    return summary


# Parent function for k-folds cross-validation
def run_kfolds(args):

    # Load training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
    num_folds = 5

    # Open hyperparameters file
    with open(args.config, "r") as f:

        # Load in hyperparams
        hyperparam_grid = yaml.safe_load(f)[args.hyperparams]

    # Print set of possible hyperparameters
    print("Hyperparameter setting ranges", hyperparam_grid)

    # Setup grid search by enumerating all possible combination of parameters in hyperparam_grid
    # e.g. {'batch_size': [24, 48], 'num_epochs': [2]} -> [{'batch_size': 24, 'num_epochs': 2}
    #                                                     {'batch_size': 48, 'num_epochs': 2}]
    keys, values = zip(*hyperparam_grid.items())
    param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create training-validation splits
    val_splits = create_stratified_kfolds(X_train, y_train, num_folds)

    # Instantiate empty list to store all parameter combinations
    pool_args = []

    # Loop through all parameters
    for p in param_list:

        # Append current model specs to grid-search list
        pool_args.append((p, X_train, y_train, n_classes, val_splits, X_test, y_test))

    # Empty list to store results of each parameter combination/model
    summary = []

    # Check if multiple processors are to be used
    if args.processes > 1:

        # Display number of processors in use
        print(f"Running with {args.processes} processors.")

        # Set up designated number of processors
        with Pool(processes=args.processes) as pool:

            # Train models and append results to list
            summary.append(pool.map(run_experiment, pool_args))

    # If only one processor is to be used
    else:

        # Loop through all parameter combinations
        for p in pool_args:

            # Train models and append results to list
            summary.append(run_experiment(p))

    # Open results file
    with open(args.results_file, 'w+') as f:

        # Write results
        json.dump(summary, f)


#
def arg_parser():

    # Create argument parser object
    parser = argparse.ArgumentParser()

    # Add arguments for parser
    # Add configuration file path
    parser.add_argument(
        "-c",
        "--config",
        default="hyperparams/config.yml",
        type=str,
        help="Config file for the experiment.",
    )

    # Add hyperparameters
    parser.add_argument(
        "-hy",
        "--hyperparams",
        default="params1",
        type=str,
        help="Name of hyperparameter set.",
    )

    # Add number of processors to use
    parser.add_argument(
        "-p", "--processors", default=1, type=int, help="If > 1 then use multiprocessing."
    )

    # Add k-folds indicator
    parser.add_argument(
        "-kf", "--kfolds", default=0, type=int, help="If 1 then run k-folds validation."
    )

    # Add random seed
    parser.add_argument(
        "-s", "--seed", default=42, type=int, help="Random seed used for experiment."
    )

    # Add results file
    parser.add_argument(
        "-r", "--results_file", default="results/test.json", type=str, help="File to write k-fold results."
    )

    # Parse arguments
    args = parser.parse_args()

    # Return parsed arguments
    return args


# Run script
if __name__ == "__main__":

    # Get all hyperparameter combinations
    args = arg_parser()

    # Record current time for model start
    start_time = time.time()

    # Set a random seed to reproduce results
    np.random.seed(args.seed)

    # Check if k-fold cross validation is to be used
    if args.kfolds:

        # Run modelling with k-folds configuration
        run_kfolds(args)

    # Check otherwise
    else:

        # Run standard train/test model
        run(args)

    # Display run time
    print(f"Run time: {np.round(time.time() - start_time, 2)}s")
