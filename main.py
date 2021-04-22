import numpy as np
import pandas as pd
import time
from datetime import datetime
import yaml
import itertools
import argparse
from multiprocessing import Pool
import json
import matplotlib.pyplot as plt
import os

from src.trainer.trainer import Trainer
from src.utils.io import load_directory
from src.network.mlp import MLP
from src.utils.ml import one_hot, create_stratified_kfolds, create_layer_sizes


def run_model(args, hyperparams, X_train, y_train, X_test, y_test, save=False):

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
    train_metrics = trainer.validation(X=X_train, y=y_train)

    print(f"Trained on params: {p}")
    print(f"Overall training set results: {train_metrics}")
    print(f"Overall test set results: {test_metrics}")

    if save:
        losses_dir = "analysis/losses"
        # Make the plot dir if it doesn't exist
        if not os.path.exists(losses_dir):
            os.makedirs(losses_dir)

        with open(
            f"{losses_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            "w",
        ) as f:
            json.dump(losses, f)

    return train_metrics, test_metrics, losses

def kfolds_experiment_verbose(args):
    params, X_train, y_train, n_classes, splits = args
    layer_sizes = create_layer_sizes(
        X_train.shape[1], n_classes, params["num_hidden"], params["hidden_size"]
    )
    # Set the activations for each network layer. e.g. if relu and 2 hidden, then our activations
    # are ["relu", "relu", None] since we don't have an activation on the final layer
    activations = [params["activation"]] * params["num_hidden"] + [None]

    # Set the dropout rate for each layer, keeping the first layer as 0
    dropout_rates = [0] + [params["dropout_rate"]] * params["num_hidden"]

    epochs = range(1, params["num_epochs"]+1)

    # Instatiate loss tracker per fold
    epoch_losses = {i: {"train_ce": 0, "val_ce": 0, "val_f1": 0, "val_acc": 0} for i in epochs}

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

    # Divide losses by the number of folds
    for col in epoch_losses[1]:
        for epoch in epochs:
            epoch_losses[epoch][col] /= params["num_epochs"]
    return epoch_losses


def kfolds_experiment(args):
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

        # Get training loss for fold when the model is in "test" model (e.g. no dropout)
        metrics["train_loss"] += trainer.validation(X=X_train_k, y=y_train_k)["loss"]
        # Extract validation loss and predicted labels
        val_results = trainer.validation(X=X_val_k, y=y_val_k)
        val_results["cv_loss"] = val_results["loss"]

        # Accumulate losses, accuracy, f1_macro
        for met in metrics:
            if met != "train_loss":
                metrics[met] += val_results[met]

    # Divide the total accumualted metrics by the number of folds
    for met in metrics:
        metrics[met] /= len(splits)

    print(f"Trained on params: {params}")
    print(f"Overall cross-validation loss: {metrics['cv_loss']}")
    summary = {**params, **metrics}
    return summary


def run_kfolds(args, hyperparams, X_train, y_train, write=True, save_epochs=False):
    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
    num_folds = 5

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    pool_args = []

    # Check whether we are training on multiple sets of hyperparams
    for p in hyperparams:
        pool_args.append((p, X_train, y_train, n_classes, splits))

    # Define the function which runs the experiment
    experiment = kfolds_experiment_verbose if save_epochs else kfolds_experiment
    summary = []
    if args.processes > 1:
        print(f"Running with {args.processes} processors.")
        with Pool(processes=args.processes) as pool:
            summary.append(pool.map(experiment, pool_args))
    else:
        for p in pool_args:
            summary.append(experiment(p))

    if write:
        # Write results to unique file
        results_file = (
            f"results/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        with open(results_file, "w+") as f:
            json.dump(summary, f)

    return summary


def get_learning_curve_data(args, hyperparams, X_train, y_train):
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

    for i in num_examples:
        X = X_train_shuffled[:i]
        y = y_train_shuffled[:i]
        # Get kfolds scores for this set of examples
        summary = run_kfolds(args, hyperparams, X, y, write=False)
        train_losses.append(summary[0]["train_loss"])
        cv_losses.append(summary[0]["cv_loss"])
    # print(train_losses)
    # print(cv_losses)
    data = {
        "num_examples": num_examples,
        "train_losses": train_losses,
        "cv_losses": cv_losses,
    }

    lc_dir = "analysis/learning_curves"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)

    with open(
        f"{lc_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        "w",
    ) as f:
        json.dump(data, f)
    return data


def plot_learning_curves(data):
    lc_dir = "analysis/learning_curves"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)
    plt.style.use("ggplot")
    plt.plot(data["num_examples"], data["train_losses"], "-o", label="Training Loss")
    plt.plot(data["num_examples"], data["cv_losses"], "-o", label="Validation Loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of examples")
    plt.ylabel("Cross Entropy loss")
    # The title will be better in latex
    # plt.title("Learning curves for best model")
    plt.savefig(
        f"{lc_dir}/plot_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )
    plt.show()


def plot_model_over_time(losses):

    epochs = [int(i) for i in losses["train"].keys()]
    # Get train losses in order of epoch
    # Note that python 3.6 onwards retains order in dictionaries
    train_values = list(losses["train"].values())
    train_ce = [d["loss"] for d in train_values]
    # train_acc = [d["accuracy"] for d in train_values]
    # train_f1 = [d["f1_macro"] for d in train_values]

    # Get test losses in order of epoch
    test_values = list(losses["test"].values())
    test_ce = [d["loss"] for d in test_values]
    test_acc = [d["accuracy"] for d in test_values]
    test_f1 = [d["f1_macro"] for d in test_values]

    lc_dir = "analysis/losses"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)

    plt.figure()
    plt.style.use("ggplot")
    # plt.plot(epochs, train_acc, "-o", label="Train Accuracy")
    plt.plot(epochs, test_acc, "-o", label="Test Accuracy")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig(
        f"{lc_dir}/accuracy_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )

    plt.figure()
    plt.style.use("ggplot")
    plt.plot(epochs, train_ce, "-o", label="Train CE loss")
    plt.plot(epochs, test_ce, "-o", label="Test CE loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig(
        f"{lc_dir}/ce_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )

    plt.figure()
    plt.style.use("ggplot")
    # plt.plot(epochs, train_f1, "-o", label="Train F1 score")
    plt.plot(epochs, test_f1, "-o", label="Test F1 score")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    plt.ylabel("F1 score (averaged)")
    plt.savefig(
        f"{lc_dir}/f1_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )


def get_ablation_data(args, hyperparams, X_train, y_train, X_test, y_test):
    """
    In this ablation analysis, we test the model specified in hyperparams with and without each
    module.
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
        cv_summary = run_kfolds(args, [new_hyperparams], X_train, y_train, write=False)
        # Start the timer to measure the training time
        start_time = time.time()
        train_summary, test_summary, _ = run_model(
            args, [new_hyperparams], X_train, y_train, X_test, y_test
        )
        losses[col]["Time"] = time.time() - start_time
        losses[col]["Train"] = train_summary["loss"]
        losses[col]["Val"] = cv_summary[0]["cv_loss"]
        losses[col]["Test"] = test_summary["loss"]

    ablation_dir = "analysis/ablations"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    with open(
        f"{ablation_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        "w",
    ) as f:
        json.dump(losses, f)
    return losses


def plot_ablation(data):
    """Plot ablation tables"""
    df = pd.DataFrame(data)

    df = df.round(3)
    df = df.rename(
        columns={
            "Without activations": "No activations",
            "With weight_decay=0.001": "wd=0.001",
            "Without momentum": "No momentum",
            "Without hidden layers": "No hidden layers",
            "Without dropout": "No dropout",
            "Without batchnorm": "No batchnorm",
            "Batched 10 epochs": "Batched",
            "SGD 10 epochs": "SGD",
        }
    )
    # Transpose to make multi row instead of multi column
    df = df.T

    df["Time"] = df["Time"].astype(int)
    ablation_dir = "analysis/ablations"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    # We plot a separate table for the SGD vs batched experiment run with 10 epochs
    df1 = df.loc[:"No batchnorm"]
    df2 = df.loc["Batched":]

    with open(f"{ablation_dir}/ablation_table1.text", "w") as f:
        f.write(df1.to_latex(index=True))
    with open(f"{ablation_dir}/ablation_table2.text", "w") as f:
        f.write(df2.to_latex(index=True))

    print(df1)
    print(df2)


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
        "-kf", "--kfolds", default=0, type=int, help="Whether to run kfolds validation"
    )
    parser.add_argument(
        "-s", "--seed", default=42, type=int, help="Random seed used for experiment"
    )
    parser.add_argument(
        "-se",
        "--save_epochs",
        default=0,
        type=int,
        help="Whether to save the loss per epoch in kfolds validation",
    )
    parser.add_argument(
        "-pe",
        "--plot_errors",
        default=0,
        type=int,
        help="Whether to plot model errors over time",
    )
    parser.add_argument(
        "-ef",
        "--errors_file",
        type=str,
        help="Name of file with saved data for plotting errors",
    )
    parser.add_argument(
        "-lc",
        "--learning_curves",
        default=0,
        type=int,
        help="Whether to plot learning curves",
    )
    parser.add_argument(
        "-lcf",
        "--learning_curves_file",
        type=str,
        help="Name of file with saved data for learning curves",
    )
    parser.add_argument(
        "-a",
        "--ablation",
        default=0,
        type=int,
        help="Whether to plot ablation analysis",
    )
    parser.add_argument(
        "-af",
        "--ablation_file",
        type=str,
        help="Name of file with saved ablation results",
    )
    args = parser.parse_args()
    return args


# Run script
if __name__ == "__main__":
    args = arg_parser()

    start_time = time.time()
    # Set a random seed to reproduce results
    np.random.seed(args.seed)

    # Import training and test data
    X_train, y_train, X_test, y_test = load_directory("data")

    # Load hyperparameters from file
    with open(args.config, "r") as f:
        hyperparams = yaml.safe_load(f)[args.hyperparams]

    # Check if hyperparam values are in a list format (used for kfolds validation and plotting)
    if isinstance(hyperparams["hidden_size"], list):
        # Setup grid search by enumerating all possible combination of parameters in hyperparams
        # e.g. {'batch_size': [24, 48], 'num_epochs': [2]} -> [{'batch_size': 24, 'num_epochs': 2}
        #                                                     {'batch_size': 48, 'num_epochs': 2}]
        keys, values = zip(*hyperparams.items())
        hyperparams = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        hyperparams = [hyperparams]

    if args.kfolds:
        run_kfolds(args, hyperparams, X_train, y_train, save_epochs=args.save_epochs)
    elif args.learning_curves:
        if args.learning_curves_file:
            # If we have a saved learning_curves file, don't generate a new one
            with open(args.learning_curves_file, "r") as f:
                data = json.load(f)
        else:
            data = get_learning_curve_data(args, hyperparams, X_train, y_train)
        plot_learning_curves(data)
    elif args.ablation:
        if args.ablation_file:
            # If we have a saved ablation file, don't generate a new one
            with open(args.ablation_file, "r") as f:
                losses = json.load(f)
        else:
            # Create ablation data
            losses = get_ablation_data(
                args, hyperparams, X_train, y_train, X_test, y_test
            )
        plot_ablation(losses)
    else:
        if args.errors_file:
            # If we have a saved errors file, don't generate a new one
            with open(args.errors_file, "r") as f:
                losses = json.load(f)
        else:
            # Create errors data
            _, _, losses = run_model(
                args, hyperparams, X_train, y_train, X_test, y_test, save=True
            )
        plot_model_over_time(losses)

    print("Run time: ", time.time() - start_time)
