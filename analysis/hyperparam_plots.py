import json
import time
import argparse
import pandas as pd


def write_hyperparams_table(data):
    df = pd.DataFrame(data)
    # Create normalised loss, accuracy, and f1_macro
    loss_norm = (df["loss"] - df["loss"].mean()) / df["loss"].std()
    acc_norm = (df["accuracy"] - df["accuracy"].mean()) / df["accuracy"].std()
    f1_macro_norm = (df["f1_macro"] - df["f1_macro"].mean()) / df["f1_macro"].std()
    # Combine metrics by taking their sum (we use a negative for loss since we aim to minimise it)
    # df["validation_score"] = acc_norm - loss_norm + f1_macro_norm

    # df = df.sort_values("validation_score", ascending=False, ignore_index=True)
    df = df.sort_values("loss", ascending=True, ignore_index=True)
    df = df[
        [
            "batch_size",
            "num_epochs",
            "learning_rate",
            "weight_decay",
            "momentum",
            "num_hidden",
            "hidden_size",
            "dropout_rate",
            "batch_normalisation",
            "loss",
            "accuracy",
            "f1_macro",
            # 'validation_score'
        ]
    ]
    df = df.rename(
        columns={
            "batch_size": "b_size",
            "num_epochs": "epochs",
            "learning_rate": "lr",
            "weight_decay": "wd",
            "momentum": "mom",
            "num_hidden": "n_h",
            "hidden_size": "h_size",
            "dropout_rate": "drp",
            "batch_normalisation": "bn",
            "loss": "loss",
            "accuracy": "acc",
            "f1_macro": "f1",
            # 'validation_score': 'score'
        }
    )
    df.index += 1
    df["loss"] = df["loss"].round(4)
    # df[['acc', 'f1', 'score']] = df[['acc', 'f1', 'score']].round(3)
    df[["acc", "f1"]] = df[["acc", "f1"]].round(3)
    df["bn"] = df["bn"].map({False: "N", True: "Y"})

    # Running manually without multiprocessing we get different results. Use these results to
    # replace the top 3 in the grid search for replicability
    # manual_runs = [
    #     {'loss': 'cross_entropy', 'optimiser': 'sgd', 'batch_size': 64, 'num_epochs': 100,
    #     'learning_rate': 0.01, 'weight_decay': 0.0001, 'momentum': 0.5, 'num_hidden': 4,
    #     'hidden_size': 128, 'activation': 'relu', 'dropout_rate': 0, 'batch_normalisation': False},
    #     {'loss': 'cross_entropy', 'optimiser': 'sgd', 'batch_size': 64, 'num_epochs': 100,
    #     'learning_rate': 0.01, 'weight_decay': 0.0001, 'momentum': 0.5, 'num_hidden': 3,
    #     'hidden_size': 128, 'activation': 'relu', 'dropout_rate': 0, 'batch_normalisation': False},
    #     {'loss': 'cross_entropy', 'optimiser': 'sgd', 'batch_size': 64, 'num_epochs': 100,
    #     'learning_rate': 0.01, 'weight_decay': 0.0001, 'momentum': 0.5, 'num_hidden': 2,
    #     'hidden_size': 128, 'activation': 'relu', 'dropout_rate': 0, 'batch_normalisation': False},
    # ]

    print(df.head(30))
    # Save the top 15 values to a latex table
    # Note that we overwrite the top 3 values with values obtained without multiprocessing.
    with open("analysis/hyperparam_table1.text", "w") as f:
        f.write(df[:15].to_latex(index=True))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rf",
        "--result_files",
        nargs="+",
        default="results/test.json",
        type=str,
        help="Result files from cross validation experiments (one or multiple)",
    )
    args = parser.parse_args()
    return args


def combine_results(all_files):
    data = []
    for filename in all_files:
        with open(filename, "r") as f:
            data.append(json.load(f)[0])
    flat_data = [item for sublist in data for item in sublist]
    return flat_data


# Run script
if __name__ == "__main__":
    start_time = time.time()
    args = arg_parser()

    data = combine_results(args.result_files)
    write_hyperparams_table(data)

    # Set a random seed to reproduce results
    # np.random.seed(args.seed)

    print("Run time: ", time.time() - start_time)