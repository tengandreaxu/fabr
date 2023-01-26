import os
import time
import numpy as np
import pandas as pd
from data.DatasetManager import DatasetsManager
from plotting.Plotter import Plotter
from models.FABR import FABR
from sklearn.linear_model import RidgeClassifier
from rf.RandomNeurons import RandomNeurons
from utils.file_handlers import load_pickle
from utils.sampling import get_demean_labels

if __name__ == "__main__":

    runs = 5
    shrinkages_list = [
        np.arange(0, 5),
        np.arange(0, 10),
        np.arange(0, 20),
        np.arange(0, 50),
    ]

    dm = DatasetsManager()

    rf_type = RandomNeurons(activation="linear")

    n_observations = 4000
    n_test = 1000

    n_features = [
        10,
        100,
        500,
        1000,
        2000,
        3000,
        5000,
        10000,
        25000,
        50000,
        100000,
    ]
    spectral_times = {}
    sklearn_times = {}
    sklearns = []
    fabrs = []
    # for run in range(runs):
    #     for shrinkages in shrinkages_list:
    #         spectral_times[len(shrinkages)] = []
    #         sklearn_times[len(shrinkages)] = []
    #         for n_feature in n_features:

    #             start = time.monotonic()
    #             x_train, y_train, x_test, y_test = dm.get_synthetic_dataset_normal_dist(
    #                 n_observations=(n_observations + n_test),
    #                 n_features=n_feature,
    #                 split_number=n_test,
    #                 number_classes=2,
    #                 seed=1234 + run,
    #             )

    #             y_train_demean, y_test_demean = get_demean_labels(y_train, y_test)

    #             y_train = y_train_demean.argmax(1)

    #             spectral_size = 1000  # n_feature
    #             if spectral_size == 100000:
    #                 spectral_size = 10000

    #             spectral = FABR(
    #                 rf_type=rf_type,
    #                 shrinkage_list=shrinkages,
    #                 small_subset_size=spectral_size,
    #                 just_use_original_features=True,
    #                 produce_betas=False,
    #                 use_numpy=True,
    #             )

    #             start = time.monotonic()
    #             spectral.fit(x_train=x_train, y_train=y_train_demean)
    #             spectral.predict(x_test)
    #             end = time.monotonic()
    #             spectral_time = end - start

    #             spectral_times[len(shrinkages)].append(spectral_time)

    #             print(f"now starting sklearn !!!!!!!!!")

    #             start = time.monotonic()
    #             for alpha in shrinkages:
    #                 print(f"sklearn for {alpha} running")
    #                 ridge = RidgeClassifier(alpha=alpha)
    #                 ridge.fit(X=x_train, y=y_train)
    #                 ridge.predict(X=x_test)
    #             end = time.monotonic()
    #             sklearn_time = end - start
    #             sklearn_times[len(shrinkages)].append(sklearn_time)

    #     spectral = pd.DataFrame(spectral_times)
    #     sklearn = pd.DataFrame(sklearn_times)
    #     spectral.index = n_features
    #     sklearn.index = n_features
    #     spectral.index.name = "index"
    #     sklearn.index.name = "index"
    #     spectral.to_pickle(
    #         f"results/sklearn_comparison/{n_observations}_fabr_times_run_{run}.pickle"
    #     )
    #     sklearn.to_pickle(
    #         f"results/sklearn_comparison/{n_observations}_sklearn_times_{run}.pickle"
    #     )
    #     fabrs.append(spectral)
    #     sklearns.append(sklearn)
    sklearns = [
        load_pickle(
            f"results/sklearn_comparison/{n_observations}_sklearn_times_{x}.pickle"
        )
        for x in range(runs)
    ]
    fabrs = [
        load_pickle(
            f"results/sklearn_comparison/{n_observations}_fabr_times_run_{x}.pickle"
        )
        for x in range(runs)
    ]

    sklearn = pd.concat(sklearns).groupby("index").mean()
    spectral = pd.concat(fabrs).groupby("index").mean()

    sklearn_std = pd.concat(sklearns).groupby("index").std()
    spectral_std = pd.concat(fabrs).groupby("index").std()

    plotter = Plotter()
    output_dir = "results/sklearn_comparison"
    plotter.plot_multiple_curves_from_dataframes_given_columns_same_legend(
        dfs=[spectral, sklearn],
        x="",
        ylabel="Training and Prediction Time (s)",
        xlabel=r"$\bf{d}$",
        labels=[
            [
                "FABR - $\|z\|=5$",
                "FABR - $\|z\|=10$",
                "FABR - $\|z\|=20$",
                "FABR - $\|z\|=50$",
            ],
            [
                "${\it sklearn}$ - $\|z\|=5$",
                "${\it sklearn}$ - $\|z\|=10$",
                "${\it sklearn}$ - $\|z\|=20$",
                "${\it sklearn}$ - $\|z\|=50$",
            ],
        ],
        linestyles=["solid", "dashed", "dashdot", "dotted"],
        file_name=os.path.join(output_dir, "spectral_vs_sklearn_speed.pdf"),
        from_index=True,
        colors=["black", "brown"],
        remove_right_spine=True,
        remove_top_spine=True,
    )

    plotter.plot_multiple_curves_from_dataframes_given_columns_same_legend(
        dfs=[spectral, sklearn],
        x="",
        ylabel="Training and Prediction Time (s)",
        xlabel=r"$\bf{d}$",
        labels=[
            [
                "FABR - $\|z\|=5$",
                "FABR - $\|z\|=10$",
                "FABR - $\|z\|=20$",
                "FABR - $\|z\|=50$",
            ],
            [
                "${\it sklearn}$ - $\|z\|=5$",
                "${\it sklearn}$ - $\|z\|=10$",
                "${\it sklearn}$ - $\|z\|=20$",
                "${\it sklearn}$ - $\|z\|=50$",
            ],
        ],
        linestyles=["solid", "dashed", "dashdot", "dotted"],
        file_name=os.path.join(output_dir, "spectral_vs_sklearn_speed.png"),
        from_index=True,
        colors=["black", "brown"],
        remove_right_spine=True,
        remove_top_spine=True,
    )
    spectral_std = spectral_std.round(2)
    sklearn_std = sklearn_std.round(2)
    spectral = spectral.round(2)
    sklearn = sklearn.round(2)
    # ***********
    # Latex table
    # ***********
    for column in spectral:
        spectral[column] = (
            spectral[column].apply(lambda x: str(x) + "s")
            + " $\pm$ "
            + spectral_std[column].apply(lambda x: str(x) + "s")
        )

    for column in sklearn:
        sklearn[column] = (
            sklearn[column].apply(lambda x: str(x) + "s")
            + " $\pm$ "
            + sklearn_std[column].apply(lambda x: str(x) + "s")
        )
    spectral = spectral.rename(
        columns={
            5: "$|z|=5$ - FABR",
            10: "$|z|=10$ - FABR",
            20: "$|z|=20$ - FABR",
            50: "$|z|=50$ - FABR",
        }
    )

    sklearn = sklearn.rename(
        columns={
            5: "$|z|=5$ - ${\it sklearn}$",
            10: "$|z|=10$ - ${\it sklearn}$",
            20: "$|z|=20$ - ${\it sklearn}$",
            50: "$|z|=50$ - ${\it sklearn}$",
        }
    )

    df = pd.merge(spectral, sklearn, on="index")
    df.index.name = "$d$"
    df = df[
        [
            "$|z|=5$ - FABR",
            "$|z|=5$ - ${\it sklearn}$",
            "$|z|=10$ - FABR",
            "$|z|=10$ - ${\it sklearn}$",
            "$|z|=20$ - FABR",
            "$|z|=20$ - ${\it sklearn}$",
            "$|z|=50$ - FABR",
            "$|z|=50$ - ${\it sklearn}$",
        ]
    ]
    df.to_latex("results/sklearn_comparison/table.tex", escape=False)
