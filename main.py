import numpy as np
import time
from datetime import datetime
import yaml
import itertools
import argparse
from multiprocessing import Pool
import json

from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import one_hot, create_stratified_kfolds, create_layer_sizes


def run(args):
    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Define the number of classes
    n_classes = len(np.unique(y_train))

    # Load hyperparameters from file
    with open(args.config, "r") as f:
        p = yaml.safe_load(f)[args.hyperparams]

    # Create a list of tuples indicating the size of each network layer
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, p["num_hidden"], p["hidden_size"]
    )

    # Set the activations for each network layer. e.g. if relu and 2 hidden, then our activations
    # are ["relu", "relu", None] since we don't have an activation on the final layer
    activations = [p["activation"]] * p["num_hidden"] + [None]

    # Set the dropout rate for each layer, keeping the first layer as 0
    dropout_rates = [0] + [p["dropout_rate"]] * p["num_hidden"]

    # Create multi-layer perceptron model (i.e build model object)
    model = MLP(
        layer_sizes=layer_sizes,
        activations=activations,
        dropout_rates=dropout_rates,
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


def run_experiment(args):
    params, X_train, y_train, n_classes, splits = args
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, params["num_hidden"], params["hidden_size"]
    )
    # Set the activations for each network layer. e.g. if relu and 2 hidden, then our activations
    # are ["relu", "relu", None] since we don't have an activation on the final layer
    activations = [params["activation"]] * params["num_hidden"] + [None]

    # Set the dropout rate for each layer, keeping the first layer as 0
    dropout_rates = [0] + [params["dropout_rate"]] * params["num_hidden"]

    # Instantiate accumulated loss tracker
    acc_loss = 0

    # Train and test on each k-fold split
    for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):
        # Define model using current architecture
        model = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            batch_normalisation=params["batch_normalisation"],
        )

        # Create trainer object to handle train each epoch (define input data and parameters)
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

    cv_loss = acc_loss / len(splits)

    print(f"Trained on params: {params}")
    print(f"Overall cross-validation loss: {cv_loss}")
    summary = {**params, "cv_loss": cv_loss}
    return summary


def run_kfolds(args):
    # Load training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
    num_folds = 5

    # Load hyperparameters from file
    with open(args.config, "r") as f:
        hyperparam_grid = yaml.safe_load(f)[args.hyperparams]
    # print("Hyperparameter setting ranges", hyperparam_grid)

    # Setup grid search by enumerating all possible combination of parameters in hyperparam_grid
    # e.g. {'batch_size': [24, 48], 'num_epochs': [2]} -> [{'batch_size': 24, 'num_epochs': 2}
    #                                                     {'batch_size': 48, 'num_epochs': 2}]
    keys, values = zip(*hyperparam_grid.items())
    param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    pool_args = []

    for p in param_list:
        pool_args.append((p, X_train, y_train, n_classes, splits))

    summary = []
    if args.processes > 1:
        print(f"Running with {args.processes} processors.")
        with Pool(processes=args.processes) as pool:
            summary.append(pool.map(run_experiment, pool_args))
    else:
        for p in pool_args:
            summary.append(run_experiment(p))

    # Write results to unique file
    results_file = (
        f"results/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    with open(results_file, "w+") as f:
        json.dump(summary, f)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="hyperparams/config.yml",
        type=str,
        help="Config file for the experiment",
    )
    parser.add_argument(
        "-hy",
        "--hyperparams",
        default="params1",
        type=str,
        help="Name of hyperparameter set",
    )
    parser.add_argument(
        "-p", "--processes", default=1, type=int, help="If > 1 then use multiprocessing"
    )
    parser.add_argument(
        "-kf", "--kfolds", default=0, type=int, help="If 1 then run kfolds validation"
    )
    parser.add_argument(
        "-s", "--seed", default=42, type=int, help="Random seed used for experiment"
    )
    args = parser.parse_args()
    return args


# Run script
if __name__ == "__main__":
    args = arg_parser()

    start_time = time.time()
    # Set a random seed to reproduce results
    np.random.seed(args.seed)

    if args.kfolds:
        run_kfolds(args)
    else:
        run(args)

    print("Run time: ", time.time() - start_time)
