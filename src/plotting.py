import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
import os
import json
from datetime import datetime


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
    plt.savefig(f"{lc_dir}/accuracy_f1_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")

    plt.figure()
    plt.style.use("ggplot")
    plt.plot(epochs, train_ce, "-o", label="Train CE loss")
    plt.plot(epochs, val_ce, "-o", label="Validation CE loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig(f"{lc_dir}/ce_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")


def plot_ablation(data, ablation_dir="analysis/ablations"):

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
        # We plot a separate table for the SGD vs mini-batched experiment run with 10 epochs
        SGD_df = sub_df.loc[:"No batchnorm"]
        mini_batchGD_df = sub_df.loc["Batched":]

        # Write data to file and display
        with open(f"{ablation_dir}/{metric}_ablation_tab1.text", "w") as f:
            f.write(SGD_df.to_latex(index=True))
        with open(f"{ablation_dir}/{metric}_ablation_tab2.text", "w") as f:
            f.write(mini_batchGD_df.to_latex(index=True))

        print(SGD_df)
        print(mini_batchGD_df)


def write_hyperparams_table(all_files, grid_dir="analysis/grid_search"):

    # Combine the results from all grid search result files
    raw_data = []
    for filename in all_files:
        with open(filename, "r") as f:
            file_data = json.load(f)
            # Handle deprected nested listed structure
            if isinstance(file_data[0], list):
                file_data = file_data[0]
            raw_data.append(file_data)
    data = [item for sublist in raw_data for item in sublist]

    df = pd.DataFrame(data)
    # Handle deprecated input format
    if pd.api.types.is_string_dtype(df["loss"]):
        df.drop(columns=["loss"], inplace=True)
        df.rename(columns={"cv_loss": "loss"}, inplace=True)

    # Create normalised loss, accuracy, and f1_macro
    loss_norm = (df["loss"] - df["loss"].mean()) / df["loss"].std()
    acc_norm = (df["accuracy"] - df["accuracy"].mean()) / df["accuracy"].std()
    f1_macro_norm = (df["f1_macro"] - df["f1_macro"].mean()) / df["f1_macro"].std()

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
        }
    )
    df.index += 1
    df["loss"] = df["loss"].round(4)
    df[["acc", "f1"]] = df[["acc", "f1"]].round(3)
    df["bn"] = df["bn"].map({False: "N", True: "Y"})

    print(df.head(20))

    # Make the gridsearch dir if it doesn't exist
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    # Save the top 15 values to a latex table
    with open(f"{grid_dir}/hyperparam_table.text", "w") as f:
        f.write(df[:15].to_latex(index=True))
