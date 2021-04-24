import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
import os
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
