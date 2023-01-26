import os
import time
from utils.file_handlers import save_pickle
import pandas as pd
from typing import Any, Optional
import numpy as np
from rf.RandomFeaturesType import RandomFeaturesType
from models.FABR import FABR
from models.FABRNu import FABRNu
from models.FABRBatch import FABRBatch
from utils.sampling import get_demean_labels, sample_balanced_trainset
from plotting.Plotter import Plotter
import logging

logging.basicConfig(level=logging.INFO)


class Runner:
    """
    The singleton in charge to run experiments
    given:

    1. Dataset
    2. Model

    """

    def __init__(self):
        self.logger = logging.getLogger("Runner")

    def __get_model(
        self,
        model_name: str,
        shrinkages,
        small_subset_size: int,
        seed: int,
        rf_type: RandomFeaturesType,
        max_multiplier: int,
        produce_voc_curve: bool,
        niu: int,
        pred_batch_size: int,
        conv_parameters: Any,
        reduce_voc_curve: int,
        batch_size: int,
        shift_random_seed: bool,
    ):
        # *******************
        # FABR
        # *******************
        if model_name == "FABR":
            model = FABR(
                rf_type=rf_type,
                shrinkage_list=shrinkages,
                small_subset_size=small_subset_size,
                debug=False,
                seed=seed,
                max_multiplier=max_multiplier,
                produce_voc_curve=produce_voc_curve,
                reduce_voc_curve=reduce_voc_curve,
                shift_random_seed=shift_random_seed,
            )

        # TODO
        # ******************
        # FABR Nu - Spectral Decomposition
        # ******************
        elif model_name == "FABRNu":
            model = FABRNu(
                rf_type,
                shrinkage_list=shrinkages,
                niu=niu,
                small_subset_size=small_subset_size,
                debug=False,
                seed=seed,
                max_multiplier=max_multiplier,
                produce_voc_curve=produce_voc_curve,
                reduce_voc_curve=reduce_voc_curve,
                shift_random_seed=shift_random_seed,
            )

        # ******************
        # FABR Batch
        # ******************
        elif model_name == "FABRBatch":
            model = FABRBatch(
                rf_type,
                shrinkage_list=shrinkages,
                small_subset_size=small_subset_size,
                debug=False,
                seed=seed,
                pred_batch_size=pred_batch_size,
                max_multiplier=max_multiplier,
                produce_voc_curve=produce_voc_curve,
                reduce_voc_curve=reduce_voc_curve,
                convolution_parameters=conv_parameters,
                batch_size=batch_size,
                shift_random_seed=shift_random_seed,
            )
        return model

    def save_results(
        self,
        shrinkages: list,
        complexities: list,
        y_test_run: np.ndarray,
        y_predictions: np.ndarray,
        seed: int,
        accuracies: pd.DataFrame,
        output_dir: str,
    ):
        save_pickle(shrinkages, "shrinkages.pickle")
        save_pickle(complexities, "complexities.pickle")
        # y_test
        y_test_name = os.path.join(output_dir, f"seed_{seed}_y_test.pickle")
        save_pickle(y_test_run, y_test_name)
        # y_hat
        y_hat_name = os.path.join(output_dir, f"seed_{seed}_y_hat.pickle")
        save_pickle(y_predictions, y_hat_name)
        save_pickle(
            accuracies, os.path.join(output_dir, f"seed_{seed}_test_accuracies.csv")
        )
        self.logger.info(f"All results saved")

    @staticmethod
    def save_tables_and_plot(
        final_accuracies: list,
        performance_table: list,
        output_dir: str,
        shrinkages: list,
        runs: int,
        suptitle_: Optional[bool] = False,
        benchmark_mean: Optional[float] = 0.0,
        benchmark_std: Optional[float] = 0.0,
        sample: Optional[bool] = 0,
        benchmark_name: Optional[str] = "",
    ):
        final_accuracies = pd.concat(final_accuracies)
        final_accuracies.index.name = "index"

        # We take the average best model performance
        # That is, we take the average of the maxes
        performance_table = pd.DataFrame(performance_table)
        best_means = performance_table.mean()
        best_means.to_csv(os.path.join(output_dir, "mean_performances.csv"))
        std_accuracies = performance_table.std()
        std_accuracies.to_csv(os.path.join(output_dir, "std_performance.csv"))

        # To Plot the curve
        mean_accuracies = final_accuracies.groupby("index").mean()
        mean_accuracies.to_csv(os.path.join(output_dir, "means.csv"))
        table = []
        labels = []

        for column, index in performance_table.items():

            std = std_accuracies.loc[column]
            mean = best_means.loc[column]
            table.append({"z": column, "complexity": index, "std": std, "mean": mean})
            mean_percent = mean * 100
            std_percent = std * 100
            labels.append(
                f"z={column} - acc={mean_percent:.2f}% $\pm$ {std_percent:.2f}"
            )
        best_of_the_sample = pd.DataFrame(table)
        best_of_the_sample.to_csv(
            os.path.join(output_dir, "best_accuracies.csv"), index=False
        )
        plotter = Plotter()
        bench_mean_percent = benchmark_mean * 100
        bench_std_percent = benchmark_std * 100

        if benchmark_mean != 0 and benchmark_std != 0:

            hline_label = (
                f"{benchmark_name}: {bench_mean_percent:.2f} +- {bench_std_percent:.2f}"
            )
        else:
            hline_label = ""

        if suptitle_:
            sup = output_dir.split("/")[-1]
        else:
            sup = ""
        for df, ylabel, file_name in [
            (mean_accuracies, "Accuracy (%)", f"accuracy_test_sample={sample}.png"),
        ]:
            plotter.plot_multiple_curves_from_dataframe_given_columns(
                df=df,
                columns=shrinkages,
                labels=labels,
                x=df.index,
                ylabel=ylabel,
                suptitle=sup,
                xlabel="$c$",
                file_name=os.path.join(output_dir, f"spectral_{file_name}"),
                from_index=True,
                vline=1,
                title=f"Mean over {runs} runs - sample={sample*10}",
                hline=benchmark_mean,
                hline_label=hline_label,
            )

    def run_experiment_with_ready_data(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        runs: int,
        model_name: str,
        rf_type: RandomFeaturesType,
        shrinkages: list,
        max_multiplier: int,
        small_subset_size: int,
        reduce_voc_curve: int,
        output_dir: str,
        shift_random_seed: Optional[bool] = False,
        pred_batch_size: Optional[int] = 0,
        conv_parameters: Optional[Any] = None,
        produce_voc_curve: Optional[bool] = True,
        suptitle_: Optional[bool] = False,
        niu: Optional[int] = 0,
        benchmark_mean: Optional[float] = 0.0,
        benchmark_std: Optional[float] = 0.0,
        sample: Optional[bool] = 0,
        benchmark_name: Optional[str] = "",
        seed: Optional[int] = 0,
        batch_size: Optional[int] = 0,
    ):
        """
        Runs the experiments
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created folder: {output_dir}")

        final_accuracies = []
        performance_table = []
        training_times = []
        # K is the the Seed
        for k in range(runs):

            # We load the entire dataset and sampel with seed as number of run
            if sample > 0:
                X_train_run, y_train_run = sample_balanced_trainset(
                    x_train, y_train, sample, k, range(10)
                )
                y_train_run, y_test_run = get_demean_labels(y_train_run, y_test)

            else:
                X_train_run = x_train
                y_train_run, y_test_run = get_demean_labels(y_train, y_test)
            logging.info(f"sample dim: {X_train_run.shape}")

            self.logger.info(f"Sample = {sample}\tP1={small_subset_size}")

            # The training time
            start = time.monotonic()
            seed = k
            model = self.__get_model(
                model_name=model_name,
                rf_type=rf_type,
                shrinkages=shrinkages,
                small_subset_size=small_subset_size,
                seed=seed,
                max_multiplier=max_multiplier,
                produce_voc_curve=produce_voc_curve,
                niu=niu,
                conv_parameters=conv_parameters,
                reduce_voc_curve=reduce_voc_curve,
                batch_size=batch_size,
                pred_batch_size=pred_batch_size,
                shift_random_seed=shift_random_seed,
            )

            model.fit(
                x_train=X_train_run,
                y_train=y_train_run,
            )
            end = time.monotonic()
            training_time = end - start
            training_times.append(
                {
                    "seed": seed,
                    "training_time": training_time,
                    "p1": small_subset_size,
                    "method": model.name,
                    "training_size": X_train_run.shape[0],
                    "number_random_features": model.features[-1],
                    "voc_grid_size": len(model.voc_grid),
                }
            )
            model.predict(x_test)
            logging.info("End Ridge")

            model.compute_accuracy(y_test=y_test_run)
            model.save_betas(output_dir)
            model.save_y_test(output_dir)
            model.save_predictions(output_dir)
            model.save_accuracies(output_dir)
            final_accuracies.append(model.accuracies)
            performance_table.append(model.accuracies.max().to_dict())

        pd.DataFrame(training_times).to_csv(
            os.path.join(output_dir, "training_times.csv"), index=False
        )
        self.save_tables_and_plot(
            final_accuracies,
            performance_table,
            output_dir,
            shrinkages,
            runs,
            suptitle_,
            benchmark_mean,
            benchmark_std,
            sample,
            benchmark_name,
        )


if __name__ == "__main__":
    voc_curve_results = "results/small_sample_voc"
    results = os.listdir()
    results.sort()

    for result in results:
        result_dir = os.path.join(voc_curve_results, result)
        final_accuracies = []
        accuracies = os.listdir(result_dir)
        final_accuracies = [
            pd.read_csv(os.path.join(result_dir, x))
            for x in accuracies
            if x.startswith("accuracies_")
        ]
