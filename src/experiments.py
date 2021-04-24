import numpy as np
from datetime import datetime
from multiprocessing import Pool
import json
import os
from datetime import datetime
import time

from src.trainer.trainer import Trainer
from src.network.mlp import MLP
from src.utils.ml import create_stratified_kfolds, create_layer_sizes


def run_basic(args, hyperparams, X_train, y_train, X_test, y_test, save=False):

    """
    Most basic experiment (primarily for testing the code-base or simple models).
    Trains and tests the model based on the hyperparameters with no cross validation.
    Returns:
        train_metrics (dict)
        test_metrics (dict)
        losses (dict)
    """

    # Define the number of classes
    n_classes = len(np.unique(y_train))

    # Since we only support training on a single set of hyperparams in this function, extract the
    # hyperparam dictionary
    p = hyperparams[0]

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

    # Train model whilst saving test metrics
    losses = trainer.train(X_test=X_test, y_test=y_test)

    # Get most recent test results, noting that we save all metrics every 5 epochs
    test_metrics = losses["test"][max(losses["test"])]

    # Get training error metrics with model in test mode (this turns off dropout among others)
    train_metrics = trainer.test(X=X_train, y=y_train)

    # Display final results
    print(f"Trained on params: {p}")
    print(f"Overall training set results: {train_metrics}")
    print(f"Overall test set results: {test_metrics}")

    # Check if losses are to be saved
    if save:
        results_dir = "results/"

        # Make the losses dir if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Write to file
        with open(
            f"{results_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            "w",
        ) as f:
            json.dump(losses, f)

    # Return metrics
    return train_metrics, test_metrics, losses


def run_kfolds(args, hyperparams, X_train, y_train, write=True, save_epochs=False):

    """
       Uses stratified k-folds for cross validation.
       Trains.
       Returns:
           train_metrics (dict)
           test_metrics (dict)
           losses (dict)
    """

    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
    num_folds = 5

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    # Store all model combinations in list
    pool_args = []
    for p in hyperparams:
        pool_args.append((p, X_train, y_train, n_classes, splits))

    # Define the function which runs the experiment
    experiment = kfolds_experiment_verbose if save_epochs else kfolds_experiment

    # Run experiment over all hyperparams
    summary = []
    if args.processes > 1:
        print(f"Running with {args.processes} processors.")
        with Pool(processes=args.processes) as pool:
            summary.append(pool.map(experiment, pool_args))
    else:
        for p in pool_args:
            summary.append(experiment(p))

    # Write results to unique file
    if write:
        results_file = (
            f"results/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        with open(results_file, "w+") as f:
            json.dump(summary, f)

    # Return summary of results
    return summary


def kfolds_experiment(args):

    # Extract arguments
    params, X_train, y_train, n_classes, splits = args
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, params["num_hidden"], params["hidden_size"]
    )

    # Set the activations for each network layer. e.g. if relu and 2 hidden, then our activations
    # are ["relu", "relu", None] since we don't have an activation on the final layer
    activations = [params["activation"]] * params["num_hidden"] + [None]

    # Set the dropout rate for each layer, keeping the first layer as 0
    dropout_rates = [0] + [params["dropout_rate"]] * params["num_hidden"]

    # Instantiate accumulated trackers of training loss, cv loss, accuracy, and f1_macro
    metrics = {"train_loss": 0, "cv_loss": 0, "accuracy": 0, "f1_macro": 0}

    # Train and test on each k-fold split
    for k, (X_train_k, y_train_k, X_val_k, y_val_k) in enumerate(splits):

        # Define model using current architecture
        model = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            batch_normalisation=params["batch_normalisation"],
        )

        # Create trainer object to train each epoch (define input data and parameters)
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

        # Get training loss for model (uses test mode)
        metrics["train_loss"] += trainer.test(X=X_train_k, y=y_train_k)["loss"]

        # Get validation loss
        val_results = trainer.test(X=X_val_k, y=y_val_k)
        val_results["cv_loss"] = trainer.test(X=X_val_k, y=y_val_k)["loss"]

        # Accumulate losses, accuracy, f1_macro for all folds
        for met in metrics:
            if met != "train_loss":
                metrics[met] += val_results[met]

    # Divide the total accumulated metrics by the number of folds
    for met in metrics:
        metrics[met] /= len(splits)

    # Display model and performance
    print(f"Trained on params: {params}")
    print(f"Overall cross-validation loss: {metrics['cv_loss']}")
    summary = {**params, **metrics}

    # Return model data
    return summary


def kfolds_experiment_verbose(args):

    # Extract arguments
    params, X_train, y_train, n_classes, splits = args
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, params["num_hidden"], params["hidden_size"]
    )

    # Set the activations for each network layer. e.g. if relu and 2 hidden, then our activations
    # are ["relu", "relu", None] since we don't have an activation on the final layer
    activations = [params["activation"]] * params["num_hidden"] + [None]

    # Set the dropout rate for each layer, keeping the first layer as 0
    dropout_rates = [0] + [params["dropout_rate"]] * params["num_hidden"]

    # Instatiate loss tracker per fold
    epochs = range(1, params["num_epochs"] + 1)
    epoch_losses = {
        i: {"train_ce": 0, "val_ce": 0, "val_f1": 0, "val_acc": 0} for i in epochs
    }

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

        # Train model on training set, returning losses for each epoch on train and val sets
        losses = trainer.train(X_test=X_val_k, y_test=y_val_k)
        for epoch in epochs:
            epoch_losses[epoch]["train_ce"] += losses["train"][epoch]["loss"]
            epoch_losses[epoch]["val_ce"] += losses["test"][epoch]["loss"]
            epoch_losses[epoch]["val_acc"] += losses["test"][epoch]["accuracy"]
            epoch_losses[epoch]["val_f1"] += losses["test"][epoch]["f1_macro"]

    # Get average losses per fold
    for col in epoch_losses[1]:
        for epoch in epochs:
            epoch_losses[epoch][col] /= len(splits)
    return epoch_losses

def get_learning_curve_data(
    args, hyperparams, X_train, y_train, lc_dir="analysis/learning_curves"
):

    # Randomly shuffle all data
    idxs = list(range(len(X_train)))
    np.random.shuffle(idxs)
    X_train_shuffled = X_train[idxs]
    y_train_shuffled = y_train[idxs]

    train_losses = []
    cv_losses = []
    increment = 5000

    # Number of examples to increase for each datapoint (e.g. 5000 -> 5000, 10000, ..., 50000)
    num_examples = list(range(increment, len(X_train) + 1, increment))

    # loop through samples and extract metrics
    for i in num_examples:
        X = X_train_shuffled[:i]
        y = y_train_shuffled[:i]
        summary = run_kfolds(args, hyperparams, X, y, write=False)
        train_losses.append(summary[0]["train_loss"])
        cv_losses.append(summary[0]["cv_loss"])

    data = {
        "num_examples": num_examples,
        "train_losses": train_losses,
        "cv_losses": cv_losses,
    }

    # Make the learning curve directory if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)

    # Write results to file
    with open(
        f"{lc_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        "w",
    ) as f:
        json.dump(data, f)

    # Return training and validation results
    return data

def get_ablation_data(
    args,
    hyperparams,
    X_train,
    y_train,
    X_test,
    y_test,
    ablation_dir="analysis/ablations",
):

    """
    Performs and ablation analysis
    """

    # We assume that hyperparams is a list of length 1 to run this function
    hyperparams = hyperparams[0]

    # Create dict to keep track of ablation losses and execution time
    cols = [
        "Best model",
        "Without activations",
        "With weight_decay=0.001",
        "Without momentum",
        "Without hidden layers",
        "Without dropout",
        "Without batchnorm",
        "Batched 10 epochs",
        "SGD 10 epochs",
    ]

    losses = {k: {} for k in cols}
    for col in cols:
        new_hyperparams = hyperparams.copy()
        if col == "With weight_decay=0.001":
            new_hyperparams["weight_decay"] = 0.001
        elif col == "Without activations":
            new_hyperparams["activation"] = None
        elif col == "Without momentum":
            new_hyperparams["momentum"] = 0
        elif col == "Without hidden layers":
            new_hyperparams["num_hidden"] = 0
        elif col == "Without dropout":
            new_hyperparams["dropout_rate"] = 0
        elif col == "Without batchnorm":
            new_hyperparams["batch_normalisation"] = False
        elif col == "Batched 10 epochs":
            new_hyperparams["num_epochs"] = 10
        elif col == "SGD 10 epochs":
            new_hyperparams["num_epochs"] = 10
            new_hyperparams["batch_size"] = 1

        # Run kfolds and train/test
        cv_summary = run_kfolds(
            args, [new_hyperparams], X_train, y_train, write=False
        )

        # Start the timer to measure the training time
        start_time = time.time()
        train_summary, test_summary, _ = run_basic(
            args, [new_hyperparams], X_train, y_train, X_test, y_test
        )
        losses[col]["Time"] = time.time() - start_time
        losses[col]["Train"] = train_summary
        losses[col]["Val"] = {
            "loss": cv_summary[0]["cv_loss"],
            "accuracy": cv_summary[0]["accuracy"],
            "f1_macro": cv_summary[0]["f1_macro"],
        }
        losses[col]["Test"] = test_summary

    # Make the plot dir if it doesn't exist
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    with open(
        f"{ablation_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        "w",
    ) as f:
        json.dump(losses, f)

    return losses