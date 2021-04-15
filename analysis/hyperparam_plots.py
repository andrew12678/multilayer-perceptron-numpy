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
    df["combined_metric"] = acc_norm - loss_norm + f1_macro_norm

    df_sorted = df.sort_values("combined_metric", ascending=False, ignore_index=True)
    print(df_sorted)
    # Save the top 15 values to a latex table
    # Note that these tables are currently too big for report (even in single column appendix)
    with open("analysis/hyperparam_table.text", "w") as f:
        f.write(df_sorted[:15].to_latex(index=False))


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
