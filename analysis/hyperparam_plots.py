import json
import time
import argparse
import pandas as pd


def write_hyperparams_table(args):
    with open(args.results_file, "r") as f:
        data = json.load(f)[0]
    df = pd.DataFrame(data)
    df = df[:15].sort_values("cv_loss", ascending=True, ignore_index=True)
    print(df)
    # Save the top 15 values to a latex table
    # Note that these tables are currently too big for report (even in single column appendix)
    with open("analysis/hyperparam_table.text", "w") as f:
        f.write(df.to_latex(index=False))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rf",
        "--results_file",
        default="results/test.json",
        type=str,
        help="Results file from cross validation experiments",
    )
    args = parser.parse_args()
    return args


# Run script
if __name__ == "__main__":
    start_time = time.time()
    args = arg_parser()
    write_hyperparams_table(args)
    # Set a random seed to reproduce results
    # np.random.seed(args.seed)

    print("Run time: ", time.time() - start_time)
