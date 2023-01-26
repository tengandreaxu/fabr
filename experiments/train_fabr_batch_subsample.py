from data.DatasetManager import DatasetsManager
from experiments.Runner import Runner
from parameters.SimpleConvolutionParameters import SimpleConvolutionParameters
from rf.RandomFeaturesType import RandomFeaturesType
from rf.RandomNeurons import RandomNeurons
from parameters.RFCifar10Parameters import RFCifar10Parameters
import logging

logging.basicConfig(level=logging.INFO)


def run_small_dataset(
    X_train,
    X_test,
    y_train,
    y_test,
    shrinkages: list,
    channels: list,
    rf_type: RandomFeaturesType,
):
    ns = [256, 512, 1024, 2048]
    arora_means = [0, 0, 0, 0, 0]

    small_subset_sizes = [32640, 32640, 32640, 32640]

    arora_stds = [0, 0, 0, 0]
    batch_sizes = [2000, 2000, 2000, 2000]
    model_name = "FABRBatch"
    for n, benchmark_mean, benchmark_std, small_subset_size, batch_size in zip(
        ns, arora_means, arora_stds, small_subset_sizes, batch_sizes
    ):

        runs = 5
        output_dir = f"results/small_sample_voc_{model_name}/cifar_{n}_{str(channels)}_{rf_type.to_string()}_c_{max_multiplier}"

        runner = Runner()

        runner.run_experiment_with_ready_data(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test,
            batch_size=batch_size,
            runs=runs,
            model_name=model_name,
            rf_type=rf_type,
            shrinkages=shrinkages,
            suptitle_=False,
            max_multiplier=max_multiplier,
            small_subset_size=small_subset_size,
            reduce_voc_curve=int(n * 10),
            output_dir=output_dir,
            produce_voc_curve=True,
            sample=n,
            benchmark_std=benchmark_std,
            benchmark_mean=benchmark_mean,
            benchmark_name="CNTK",
        )


if __name__ == "__main__":
    """ """

    shrinkages = RFCifar10Parameters.shrinkages
    max_multiplier = 15
    channels = RFCifar10Parameters.channels
    global_average_pooling = True

    dm = DatasetsManager()
    logging.info(f"Loading CIFAR")
    conv_parameters = SimpleConvolutionParameters(
        channels=channels,
        global_average_pooling=global_average_pooling,
        batch_norm=True,
    )
    # Fix Seed
    X_train, X_test, y_train, y_test = dm.get_dataset(
        dataset_name="cifar10", conv_parameters=conv_parameters, random_state=0
    )

    activation = "relu"
    rf_type = RandomNeurons(activation, gamma=[1.0, 1.0])

    run_small_dataset(X_train, X_test, y_train, y_test, shrinkages, channels, rf_type)
