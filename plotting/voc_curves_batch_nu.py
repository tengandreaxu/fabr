import os
import pandas as pd
from plotting.Plotter import Plotter

if __name__ == "__main__":
    voc_curve_results_nu = "results/small_sample_voc_FABRBatch"
    voc_curve_results_nu_full = "results/FABRNu/cifar10/"
    voc_curve_results_batch_full = "results/FABRBatch/cifar10/"
    voc_curve_results_batch = "results/small_sample_voc_FABRNu"
    for voc_curve_results in [
        voc_curve_results_batch,
        voc_curve_results_nu,
        voc_curve_results_nu_full,
        voc_curve_results_batch_full,
    ]:
        results = os.listdir(voc_curve_results)
        results.sort()

        columns = [
            "1e-05",
            "0.1",
            "1.0",
            "10.0",
            "100.0",
            "1000.0",
            "10000.0",
            "100000.0",
        ]
        labels = [
            "$z=10^{-5}$",
            "$z=10^{-1}$",
            "$z=10^{0}$",
            "$z=10^{1}$",
            "$z=10^{2}$",
            "$z=10^{3}$",
            "$z=10^{4}$",
            "$z=10^{5}$",
        ]
        plotter = Plotter()
        for result in results:
            result_dir = os.path.join(voc_curve_results, result)
            means_file = os.path.join(result_dir, "means.csv")
            if not os.path.exists(means_file):
                continue
            means = pd.read_csv(means_file)

            means.index = means["index"]
            means = means[means.index <= 15]

            means.pop("index")
            try:
                subsample = result_dir.split("/")[2].split("_")[1]
            except:
                subsample = 5000
                means["0.1"] = means["1.0"]
            prefix = "batch" if "FABRBatch" in voc_curve_results else "nu"

            for format in ["pdf", "png"]:
                plotter.plot_multiple_curves_from_dataframe_given_columns(
                    df=means,
                    columns=columns,
                    labels=labels,
                    x=means.index,
                    ylabel="Accuracy (%)",
                    suptitle="",
                    xlabel=r"$\bf{c}$",
                    file_name=os.path.join(
                        result_dir, f"{prefix}_{subsample}_voc_curve.{format}"
                    ),
                    from_index=True,
                    vline=1,
                    title=f"",
                    remove_right_spine=True,
                    remove_top_spine=True,
                )
