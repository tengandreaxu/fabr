import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from utils.printing import print_header
from utils.measures import mean_max_accuracies

if __name__ == "__main__":
    results_dir_subsample = "results/small_sample_voc/cifar_{}_[64, 256, 1024, 8192]_neurons_relu_gaussian_mixture_[1.0, 1.0]_c_25/"
    resnet_dir_resnet = "results/resnet_on_cifar/{}_batch=32_lr=0.001/"
    results_dir_full = "results/FABR/cifar10/[64, 256, 1024, 8192]_neurons_relu_gaussian_mixture_[1.0, 1.0]_c_15/"
    # We get the best FABR's up to c=15
    max_complexity = 15.0
    subsamples = [256, 512, 1024, 2048, 5000]
    shrinkages = [1.0, 100, 10000.0, 100000.0]
    table = {}
    runs = 5

    for model, folder in [
        ("ResNet-34", resnet_dir_resnet),
        ("z={}", results_dir_subsample),
    ]:
        for subsample in subsamples:
            if subsample == 256:
                table["n"] = [subsample * 10]
            else:
                table["n"].append(subsample * 10)

            print_header(f"Subsample:\t{subsample}")
            max_accuracies = []

            if model == "ResNet-34":
                result_dir = folder.format(subsample)

                for run in range(runs):
                    df = pd.read_csv(
                        os.path.join(result_dir, f"seed_{run}_test_accuracies.csv")
                    )
                    test_accuracy = df.test_accuracy.max()
                    max_accuracies.append(test_accuracy)
                mean = round(np.mean(max_accuracies), 2)
                std = round(np.std(max_accuracies), 2)
                if model in table:
                    table[model].append(f"{mean}\% $\pm$ {std:.2f}\%")
                else:
                    table[model] = [f"{mean}\% $\pm$ {std:.2f}\%"]
            else:

                if subsample != 5000:
                    result_dir = folder.format(subsample)
                else:
                    result_dir = results_dir_full
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
                    key_ = f"z={shrinkage}"

                    if key_ in table:
                        table[key_].append(f"{mean}\% $\pm$ {std}\%")
                    else:
                        table[key_] = [f"{mean}\% $\pm$ {std}\%"]

    table = pd.DataFrame(table)
    print(tabulate(table, headers="keys", tablefmt="psql"))
    print(table.to_latex(index=False, escape=False))
