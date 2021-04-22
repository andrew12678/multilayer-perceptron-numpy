import yaml
import itertools

from src.utils.io import load_directory
from src.utils.logging import *


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

    # Check if kfolds experiment and what logging is to be conducted
    if args.kfolds:
        if args.plot_errors:

            # Check if error file already exists
            if args.error_file:
                with open(args.error_file, "r") as f:
                    losses = json.load(f)
                    print("losses", losses)
            else:
                losses = run_kfolds(
                    args, hyperparams, X_train, y_train, save_epochs=True
                )

            # Plot model
            plot_model_over_time(losses, args)

        # If not plotting errors, run kfolds experiment
        else:
            run_kfolds(args, hyperparams, X_train, y_train, save_epochs=False)

    # Check if learning curve is required
    elif args.learning_curves:
        if args.learning_curves_file:
            # If we have a saved learning_curves file, don't generate a new one
            with open(args.learning_curves_file, "r") as f:
                data = json.load(f)
        else:
            data = get_learning_curve_data(args, hyperparams, X_train, y_train)
        plot_learning_curves(data, args)

    # Check if ablation study is to be performed
    elif args.ablation:
        if args.ablation_file:
            # If we have a saved ablation file, don't generate a new one
            with open(args.ablation_file, "r") as f:
                losses = json.load(f)
        else:
            # Get ablation data
            losses = get_ablation_data(
                args, hyperparams, X_train, y_train, X_test, y_test
            )
        plot_ablation(losses, args)

    # Run a basic train/test model if no other experiments selected
    else:
        run_basic(args, hyperparams, X_train, y_train, X_test, y_test, save=True)

    print("Run time: ", time.time() - start_time)
