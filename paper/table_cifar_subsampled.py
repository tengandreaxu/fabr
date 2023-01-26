import pandas as pd
from tabulate import tabulate
from utils.printing import print_header
from utils.measures import mean_max_accuracies

if __name__ == "__main__":
    results_dir = "results/small_sample_voc/cifar_{}_[64, 256, 1024, 8192]_neurons_relu_gaussian_mixture_[1.0, 1.0]_c_100"

    subsamples = [1, 2, 4, 8, 16, 32, 64, 128]
    shrinkages = [1.0, 100, 10000.0, 100000.0]
    runs = 20
    table = {}
    for subsample in subsamples:
        if subsample == 1:
            table["n"] = [subsample * 10]
        else:
            table["n"].append(subsample * 10)
        print_header(f"Subsample:\t{subsample}")

        result_dir = results_dir.format(subsample)
        mean_max_accuracy, std_max_accuracy = mean_max_accuracies(
            runs=runs, result_dir=result_dir
        )
        for shrinkage in shrinkages:

            mean_shrinkage = mean_max_accuracy[
                mean_max_accuracy.shrinkage == shrinkage
            ]["mean"].iloc[0]
            std_shrinakge = std_max_accuracy[std_max_accuracy.shrinkage == shrinkage][
                "std"
            ].iloc[0]
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
