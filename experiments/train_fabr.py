from data.DatasetManager import DatasetsManager
from experiments.Runner import Runner
from parameters.FABRArguments import FABRArguments
import logging

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    """This script runs FABR on the declared dataset and complexity"""
    fabr_arguments = FABRArguments()
    fabr_arguments.print_parameters()
    dm = DatasetsManager()

    X_train, X_test, y_train, y_test = dm.get_dataset(
        dataset_name=fabr_arguments.dataset,
        conv_parameters=fabr_arguments.conv_parameters
        if (not fabr_arguments.flatten and fabr_arguments.dataset != "mnist1m")
        else None,
        random_state=0,
        flatten=fabr_arguments.flatten,
    )

    print(X_train.shape)
    runner = Runner()
    print(fabr_arguments.sample)
    runner.run_experiment_with_ready_data(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        niu=fabr_arguments.nu,
        runs=fabr_arguments.runs,
        model_name=fabr_arguments.model_name,
        rf_type=fabr_arguments.rf_type,
        shrinkages=fabr_arguments.shrinkages,
        suptitle_=False,
        shift_random_seed=fabr_arguments.shift_random_seed,
        conv_parameters=fabr_arguments.conv_parameters
        if not fabr_arguments.batch_no_convolution
        else None,
        max_multiplier=fabr_arguments.max_multiplier,
        small_subset_size=fabr_arguments.p1,
        pred_batch_size=fabr_arguments.pred_batch_size,
        reduce_voc_curve=fabr_arguments.reduce_voc_curve,
        output_dir=fabr_arguments.results_dir,
        batch_size=fabr_arguments.batch_size,
        sample=fabr_arguments.sample,  # Full Sample
        benchmark_std=0,
        benchmark_mean=0,
        produce_voc_curve=fabr_arguments.produce_voc_curve,
        benchmark_name="",
    )
