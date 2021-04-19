import numpy as np
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


def run(args, hyperparams, X_train, y_train, X_test, y_test):

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

    # Train model
    _ = trainer.train()
    train_metrics = trainer.validation(X=X_train, y=y_train)

    # Test model
    test_metrics = trainer.validation(X=X_test, y=y_test)
    print(f"Trained on params: {p}")
    print(f"Overall training set results: {train_metrics}")
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
        regularized_train_loss = trainer.train()

        # Kill search if training loss is nan
        if np.isnan(regularized_train_loss):
            return {
                **params,
                "train_loss": np.nan,
                "cv_loss": np.nan,
                "accuracy": np.nan,
                "f1_macro": np.nan,
            }

        # Get training loss when the model is in "test" model (e.g. no dropout)
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


def run_kfolds(args, hyperparams, X_train, y_train, write=True):
    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Define number of cross-validation folds
    num_folds = 5

    # # Load hyperparameters from file
    # with open(args.config, "r") as f:
    #     hyperparam_grid = yaml.safe_load(f)[args.hyperparams]
    # # print("Hyperparameter setting ranges", hyperparam_grid)

    # # Setup grid search by enumerating all possible combination of parameters in hyperparam_grid
    # # e.g. {'batch_size': [24, 48], 'num_epochs': [2]} -> [{'batch_size': 24, 'num_epochs': 2}
    # #                                                     {'batch_size': 48, 'num_epochs': 2}]
    # keys, values = zip(*hyperparam_grid.items())
    # param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create training-validation splits
    splits = create_stratified_kfolds(X_train, y_train, num_folds)

    pool_args = []

    # Check whether we are training on multiple sets of hyperparams
    for p in hyperparams:
        pool_args.append((p, X_train, y_train, n_classes, splits))

    summary = []
    if args.processes > 1:
        print(f"Running with {args.processes} processors.")
        with Pool(processes=args.processes) as pool:
            summary.append(pool.map(run_experiment, pool_args))
    else:
        for p in pool_args:
            summary.append(run_experiment(p))

    if write:
        # Write results to unique file
        results_file = (
            f"results/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        with open(results_file, "w+") as f:
            json.dump(summary, f)

    return summary


def plot_learning_curves(args, hyperparams, X_train, y_train):
    plot_dir = "analysis/plots"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # Randomly shuffle all data
    idxs = list(range(len(X_train)))
    np.random.shuffle(idxs)
    X_train_shuffled = X_train[idxs]
    y_train_shuffled = y_train[idxs]

    train_losses = []
    cv_losses = []
    increment = 10000
    num_examples = list(range(increment, len(X_train) + 1, increment))

    # Number of examples to increase for each datapoint (e.g. 5000 -> 5000, 10000, ..., 50000)
    # increment = 10000
    # for i in num_examples:
    #     X = X_train_shuffled[:i]
    #     y = y_train_shuffled[:i]
    #     # Get kfolds scores for this set of examples
    #     summary = run_kfolds(args, hyperparams, X, y, write=False)
    #     train_losses.append(summary[0]["train_loss"])
    #     cv_losses.append(summary[0]["cv_loss"])
    # print(train_losses)
    # print(cv_losses)

    # Example losses
    train_losses = [
        0.15476380272853724,
        0.14724771435319572,
        0.14368380648098447,
        0.14226972405205962,
        0.1410810492802653,
        0.1396173106561778,
        0.13858820971661207,
        0.1376052818842732,
        0.13726238212764735,
        0.13726352644214462,
    ]
    cv_losses = [
        0.17368803855208034,
        0.16148927089504553,
        0.15650455458394166,
        0.15298698865546764,
        0.1503882746192336,
        0.14806578619777597,
        0.14688472539691927,
        0.14503313948724825,
        0.1441868445946898,
        0.1441555267272751,
    ]

    plt.style.use("ggplot")
    plt.plot(num_examples, train_losses, "-o", label="Training Loss")
    plt.plot(num_examples, cv_losses, "-o", label="Validation Loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of examples")
    plt.ylabel("Cross Entropy loss")
    # The title will be better in latex
    # plt.title("Learning curves for best model")
    plt.savefig(
        f"{plot_dir}/lc_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )
    plt.show()


def get_ablation_data(args, hyperparams, X_train, y_train):
    """
    In this ablation analysis, we run the following for each hyperparameter:
    1. Select a hyperparameter to vary and plot
    2. Fix all other hyperparameters using those from the best model
    3. Make a plot showing how the errors change overtime w.r.t the selected hyperparameter
    """
    # We assume that hyperparams is a list of length 1 to run this function
    hyperparams = hyperparams[0]

    # Load ablation hyperparameters to search over
    with open(args.config, "r") as f:
        abl_hyperparams = yaml.safe_load(f)[args.ablation_hyperparams]

    ablation_params = [
        "batch_size",
        "weight_decay",
        "momentum",
        "num_hidden",
        "dropout_rate",
        "batch_normalisation",
    ]
    # Create nested dict to keep track of ablation losses
    losses = {k: {"Without Module": {}, "With Module": {}} for k in ablation_params}
    # Iterate through all possible options for the hyperparameter
    for param in ablation_params:
        for module_status, val in zip(
            ["Without Module", "With Module"], abl_hyperparams[param]
        ):
            # Create new dictionary of hyperparams with all values from hyperparams except for val
            new_hyperparams = [{**hyperparams, param: val}]
            summary = run_kfolds(args, new_hyperparams, X_train, y_train, write=False)
            losses[param][module_status]["Train"] = summary[0]["train_loss"]
            losses[param][module_status]["Val"] = summary[0]["cv_loss"]

    ablation_dir = "analysis/ablations"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    with open(
        f"{ablation_dir}/{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        "w",
    ) as f:
        json.dump(losses, f)


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
        "-lc",
        "--learning_curves",
        default=0,
        type=int,
        help="Whether to plot learning curves",
    )
    parser.add_argument(
        "-a",
        "--ablation",
        default=0,
        type=int,
        help="Whether to plot ablation analysis",
    )
    parser.add_argument(
        "-ahy",
        "--ablation_hyperparams",
        default="grid-ablation",
        type=str,
        help="Name of grid for ablation analysis",
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
        run_kfolds(args, hyperparams, X_train, y_train)
    elif args.learning_curves:
        plot_learning_curves(args, hyperparams, X_train, y_train)
    elif args.ablation:
        get_ablation_data(args, hyperparams, X_train, y_train)
    else:
        run(args, hyperparams, X_train, y_train, X_test, y_test)

    print("Run time: ", time.time() - start_time)
