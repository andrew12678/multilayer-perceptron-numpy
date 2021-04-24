import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import os
import json
from datetime import datetime

from src.utils import experiments as exps


def get_learning_curve_data(args, hyperparams, X_train, y_train, lc_dir="analysis/learning_curves"):

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
        summary = exps.run_kfolds(args, hyperparams, X, y, write=False)
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


def plot_learning_curves(data, args, lc_dir="analysis/learning_curves"):

    # Make the learning curve plot directory if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)

    # Create plot and save to directory
    plt.style.use("ggplot")
    plt.plot(data["num_examples"], data["train_losses"], "-o", label="Training Loss")
    plt.plot(data["num_examples"], data["cv_losses"], "-o", label="Validation Loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of examples")
    plt.ylabel("Cross Entropy loss")

    plt.savefig(
        f"{lc_dir}/plot_{args.hyperparams}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )
    plt.show()


def plot_model_over_time(losses, args):

    # Assume data is a list with a single element
    losses = losses[0]

    epochs = [int(i) for i in losses]
    # Get list of errors in order of epoch
    metric_set = list(losses.values())
    # Get train losses in order of epoch
    # Note that python 3.6 onwards retains order in dictionaries
    train_ce = [d["train_ce"] for d in metric_set]

    # Get test losses in order of epoch
    val_ce = [d["val_ce"] for d in metric_set]
    val_acc = [d["val_acc"] for d in metric_set]
    val_f1 = [d["val_f1"] for d in metric_set]

    lc_dir = "analysis/losses"
    # Make the plot dir if it doesn't exist
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)

    plt.figure()
    plt.style.use("ggplot")
    plt.plot(epochs, val_acc, "-o", label="Validation Accuracy")
    plt.plot(epochs, val_f1, "-o", label="Validation F1 Score")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    # plt.ylabel("Accuracy")
    plt.savefig(
        f"{lc_dir}/accuracy_f1_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )

    plt.figure()
    plt.style.use("ggplot")
    plt.plot(epochs, train_ce, "-o", label="Train CE loss")
    plt.plot(epochs, val_ce, "-o", label="Validation CE loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig(
        f"{lc_dir}/ce_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )


def get_ablation_data(args, hyperparams, X_train, y_train, X_test, y_test, ablation_dir="analysis/ablations"):

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
        cv_summary = exps.run_kfolds(args, [new_hyperparams], X_train, y_train, write=False)

        # Start the timer to measure the training time
        start_time = time.time()
        train_summary, test_summary, _ = exps.run_basic(
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


def plot_ablation(data, ablation_dir = "analysis/ablations"):

    """Plot ablation tables"""

    # Create dataframe of metric data
    ablation_df = pd.DataFrame(data)
    ablation_df = ablation_df.round(3)
    ablation_df = ablation_df.rename(
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
    ablation_df = ablation_df.T
    ablation_df["Time"] = ablation_df["Time"].astype(int)
    ablation_df = ablation_df.rename(columns={"Time": "Time(s)"})

    # Make the plot dir if it doesn't exist
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    # Create a separate df for each evaluation metric
    for metric in data["Best model"]["Train"]:
        sub_df = copy.deepcopy(ablation_df)
        sub_df["Train"] = ablation_df["Train"].apply(lambda x: x.get(metric))
        sub_df["Val"] = ablation_df["Val"].apply(lambda x: x.get(metric))
        sub_df["Test"] = ablation_df["Test"].apply(lambda x: x.get(metric))

        sub_df = sub_df.round(3)
        # We plot a separate table for the SGD vs batched experiment run with 10 epochs
        SGD_df = sub_df.loc[:"No batchnorm"]
        BatchGD_df = sub_df.loc["Batched":]

        # Write data to file and display
        with open(f"{ablation_dir}/{metric}_ablation_tab1.text", "w") as f:
            f.write(SGD_df.to_latex(index=True))
        with open(f"{ablation_dir}/{metric}_ablation_tab2.text", "w") as f:
            f.write(BatchGD_df.to_latex(index=True))

        print(SGD_df)
        print(BatchGD_df)