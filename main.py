import yaml
import itertools
import time
import json
import numpy as np
import argparse

from src.utils.io import load_directory
from src import plotting, experiments as exps


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
        "-pe",
        "--plot_errors",
        default=0,
        type=int,
        help="Whether to plot model cross validation errors over time",
    )
    parser.add_argument(
        "-ef",
        "--error_file",
        type=str,
        help="Name of file with saved kfolds data for plotting errors",
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
    parser.add_argument(
        "-gf",
        "--grid_files",
        nargs="+",
        type=str,
        help="Result files from cross validation experiments (one or multiple)",
    )
    args = parser.parse_args()
    return args


# Run script
if __name__ == "__main__":
    args = arg_parser()

    # Get start time and set random seed for reproducibility
    start_time = time.time()
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

    # Individual model
    else:
        hyperparams = [hyperparams]

    # Check if kfolds experiment and what plotting is to be conducted
    if args.kfolds:
        if args.plot_errors:

            # Check if error file already exists
            if args.error_file:
                with open(args.error_file, "r") as f:
                    losses = json.load(f)
                    print("losses", losses)
            else:
                losses = exps.run_kfolds(
                    args, hyperparams, X_train, y_train, save_epochs=True
                )

            # Plot model
            plotting.plot_model_over_time(losses, args)

        # If not plotting errors, run kfolds experiment
        else:
            exps.run_kfolds(args, hyperparams, X_train, y_train, save_epochs=False)

    # Check if learning curve is required
    elif args.learning_curves:
        if args.learning_curves_file:
            # If we have a saved learning_curves file, don't generate a new one
            with open(args.learning_curves_file, "r") as f:
                data = json.load(f)
        else:
            data = exps.get_learning_curve_data(args, hyperparams, X_train, y_train)
        plotting.plot_learning_curves(data, args)

    # Check if ablation study is to be performed
    elif args.ablation:
        if args.ablation_file:
            # If we have a saved ablation file, don't generate a new one
            with open(args.ablation_file, "r") as f:
                losses = json.load(f)
        else:
            # Get ablation data
            losses = exps.get_ablation_data(
                args, hyperparams, X_train, y_train, X_test, y_test
            )
        plotting.plot_ablation(losses)

    # Create latex tables presenting results of grid search
    elif args.grid_files:
        plotting.write_hyperparams_table(args.grid_files)

    # Run a basic train/test model if no other experiments selected
    else:
        exps.run_basic(args, hyperparams, X_train, y_train, X_test, y_test, save=True)

    print("Run time: ", time.time() - start_time)
