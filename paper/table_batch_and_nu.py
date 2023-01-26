import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from utils.printing import print_header
from utils.measures import mean_max_accuracies

if __name__ == "__main__":
    results_dir_sub = "results/small_sample_voc_{}/cifar_{}_[64, 256, 1024, 8192]_neurons_relu_gaussian_mixture_[1.0, 1.0]_c_25/"
    result_dir_full = "results/{}/cifar10/{}_2000_[64, 256, 1024, 8192]_neurons_relu_gaussian_mixture_[1.0, 1.0]_c_15"
    max_complexity = 15.0
    subsamples = [256, 512, 1024, 2048, 5000]
    shrinkages = [1.0, 100, 10000.0, 100000.0]
    table = {}
    runs = 5

    for model in ["FABRBatch", "FABRNu"]:
        for subsample in subsamples:
            if subsample == 256:
                table["n"] = [subsample * 10]
            else:
                table["n"].append(subsample * 10)

            print_header(f"Subsample:\t{subsample}")
            max_accuracies = []

            if subsample == 5000:
                prefix = "batch" if model == "FABRBatch" else "nu"
                result_dir = result_dir_full.format(model, prefix)
            else:
                result_dir = results_dir_sub.format(model, subsample)

            mean_max_accuracy, std_max_accuracy = mean_max_accuracies(
                runs=runs, result_dir=result_dir, max_complexity=max_complexity
            )

            for shrinkage in shrinkages:

                mean_shrinkage = mean_max_accuracy[
                    mean_max_accuracy.shrinkage == shrinkage
                ]["mean"].iloc[0]
                std_shrinakge = std_max_accuracy[
                    std_max_accuracy.shrinkage == shrinkage
                ]["std"].iloc[0]
                mean = round(mean_shrinkage * 100, 2)
                std = round(std_shrinakge * 100, 2)
                key_ = f"{model} z={shrinkage}"

                if key_ in table:
                    table[key_].append(f"{mean}\% $\pm$ {std}\%")
                else:
                    table[key_] = [f"{mean}\% $\pm$ {std}\%"]

    table = pd.DataFrame(table)
    table = table[
        [
            "n",
            "FABRBatch z=1.0",
            "FABRNu z=1.0",
            "FABRBatch z=100",
            "FABRNu z=100",
            "FABRBatch z=10000.0",
            "FABRNu z=10000.0",
            "FABRBatch z=100000.0",
            "FABRNu z=100000.0",
        ]
    ]
    print(tabulate(table, headers="keys", tablefmt="psql"))
    print(table.to_latex(index=False, escape=False))
