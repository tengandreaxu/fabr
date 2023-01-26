import os
import argparse
from parameters.SimpleConvolutionParameters import SimpleConvolutionParameters
from rf.RandomNeurons import RandomNeurons

import logging

logging.basicConfig(level=logging.INFO)


class FABRArguments:
    def __init__(self):
        self.logger = logging.getLogger("FABRArguments")
        self.dataset = None
        self.max_multiplier = None
        self.runs = None
        self.p1 = None
        self.use_original_features = None
        self.voc_curve = None
        self.pred_batch_size = None
        self.batch_size = None
        self.sample = 0
        self.parse_arguments()
        self.set_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", dest="dataset", type=str)
        parser.set_defaults(dataset="mnist")

        parser.add_argument("--model-name", dest="model_name", type=str)
        parser.set_defaults(model_name="FABR")

        parser.add_argument("--max-multiplier", dest="max_multiplier", type=int)
        parser.set_defaults(max_multiplier=20)

        parser.add_argument("--multiplier", dest="multiplier", type=float)
        parser.set_defaults(multiplier=0.0)

        parser.add_argument("--runs", dest="runs", type=int)
        parser.set_defaults(runs=20)

        parser.add_argument("--p1", dest="p1", type=int)
        parser.set_defaults(small_subset_size=100000)

        parser.add_argument(
            "--use-original-features", dest="use_original_features", action="store_true"
        )
        parser.set_defaults(just_use_original_features=False)

        parser.add_argument(
            "--voc-curve", dest="produce_voc_curve", action="store_true"
        )
        parser.set_defaults(produce_voc_curve=False)

        parser.add_argument("--pred-batch-size", dest="pred_batch_size", type=int)
        parser.set_defaults(prediction_batch_size=0)

        parser.add_argument("--batch-size", dest="batch_size", type=int)
        parser.set_defaults(batch_size=None)

        parser.add_argument("--flatten", dest="flatten", action="store_true")
        parser.set_defaults(flatten=False)

        parser.add_argument(
            "--shrinkages",
            help="comma delimited list of shrinkages",
            dest="conv_channels",
            type=lambda s: [float(item) for item in s.split(",")],
        )
        parser.set_defaults(
            shrinkages=[
                0.00001,
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
                5000,
                10000,
                15000,
                20000,
                100000,
                1000000,
            ]
        )

        parser.add_argument(
            "--conv-channels",
            help="comma delimited list of convolution layers",
            dest="conv_channels",
            type=lambda s: [float(item) for item in s.split(",")],
        )
        parser.set_defaults(
            conv_channels=[
                64,
                256,
                1024,
                8192,
            ]
        )
        parser.add_argument("--nu", dest="nu", type=int)
        parser.set_defaults(nu=0)

        parser.add_argument("--activation", dest="activation", type=str)
        parser.set_defaults(activation="relu")

        parser.add_argument(
            "--global-average-pooling", "-gap", dest="gap", action="store_true"
        )
        parser.set_defaults(gap=False)

        parser.add_argument(
            "--no-convolution", dest="no_convolution", action="store_true"
        )
        parser.set_defaults(no_convolution=False)

        parser.add_argument("--sample", dest="sample", type=int)
        parser.set_defaults(sample=5000)

        parser.add_argument("--reduce-voc-curve", dest="reduce_voc_curve", type=int)
        parser.set_defaults(reduce_voc_curve=0)

        parser.add_argument(
            "--batch-no-convolution", dest="batch_no_convolution", action="store_true"
        )
        parser.set_defaults(batch_no_convolution=False)

        parser.add_argument(
            "--no-batch-normalization",
            dest="no_batch_normalization",
            action="store_true",
        )
        parser.set_defaults(no_batch_normalization=False)

        parser.add_argument(
            "--shift-random-seed",
            dest="shift_random_seed",
            action="store_true",
            help="""When we fit the entire dataset and we want to estimate
             uncertainty, we usually run the same training several times. 
             This will shift the random features generation seed by the number 
             of the run""",
        )
        parser.set_defaults(shift_random_seed=False)
        self.args = parser.parse_args()

    def set_arguments(self):

        self.dataset = self.args.dataset
        assert self.dataset in [
            "mnist",
            "fmnist",
            "cifar10",
            "mnist1m",
            "mnist8m",
            "mnist67m",
            "tinyimagenet",
        ], f"The dataset {self.dataset} is not available"

        self.model_name = self.args.model_name
        assert self.model_name in [
            "FABR",
            "FABRNu",
            "FABRBatch",
        ], f"The model name {self.model_name} does not exist!"

        self.max_multiplier = self.args.max_multiplier
        if self.args.multiplier > 0:
            self.max_multiplier = self.args.multiplier

        self.runs = self.args.runs
        self.p1 = self.args.p1
        self.reduce_voc_curve = self.args.reduce_voc_curve
        self.use_original_features = self.args.use_original_features
        self.pred_batch_size = self.args.pred_batch_size
        self.batch_size = self.args.batch_size
        self.shrinkages = self.args.shrinkages
        self.batch_no_convolution = self.args.batch_no_convolution
        self.sample = self.args.sample
        self.flatten = self.args.flatten
        self.shift_random_seed = self.args.shift_random_seed
        self.produce_voc_curve = self.args.produce_voc_curve
        self.conv_parameters = SimpleConvolutionParameters(
            channels=self.args.conv_channels,
            global_average_pooling=self.args.gap,
            batch_norm=not self.args.no_batch_normalization,
        )

        self.nu = self.args.nu
        if self.args.no_convolution:
            self.conv_parameters = None
        self.activation = self.args.activation
        self.rf_type = RandomNeurons(activation=self.activation)

        if self.model_name == "FABRNu":
            self.results_dir = f"results/{self.model_name}/{self.dataset}/nu_{self.nu}_{str(self.conv_parameters.channels)}_{self.rf_type.to_string()}_c_{self.max_multiplier}"
        elif self.model_name == "FABRBatch":
            self.results_dir = f"results/{self.model_name}/{self.dataset}/batch_{self.batch_size}_{str(self.conv_parameters.channels)}_{self.rf_type.to_string()}_c_{self.max_multiplier}"
        else:
            self.results_dir = f"results/{self.model_name}/{self.dataset}/{str(self.conv_parameters.channels)}_{self.rf_type.to_string()}_c_{self.max_multiplier}"

        if self.use_original_features or self.flatten:
            self.conv_parameters = None
            self.results_dir = f"results/{self.model_name}/{self.dataset}/batch_{self.batch_size}_flatten_{self.rf_type.to_string()}_c_{self.max_multiplier}"

        os.makedirs(self.results_dir, exist_ok=True)

    def print_parameters(self):

        self.logger.info(f"\tModel Running:{self.model_name}")
        self.logger.info(f"\tDataset:{self.dataset}")
        self.logger.info(f"\tMaxComplexity:{self.max_multiplier}")
        self.logger.info(f"\tP1:{self.p1}")

        if self.model_name == "FABRBatch":
            self.logger.info(f"\tBatchSize:{self.batch_size}")
            self.logger.info(f"\tPredBatchSize:{self.pred_batch_size}")

        if self.model_name == "FABRNu":
            self.logger.info(f"\tNu:{self.nu}")

        self.logger.info(f"\tRFType:{self.rf_type.to_string()}")
        self.logger.info(f"\tRuns:{self.runs}")
        self.logger.info(f"\tProduceVoCCurve:{self.produce_voc_curve}")
        self.logger.info(f"\tReduceVoCCurve:{self.reduce_voc_curve}")
        self.logger.info(f"\tSample:{self.sample*10}")
        self.logger.info(f"\tShift Random Seed: {self.shift_random_seed}")
        if self.conv_parameters is not None:
            self.logger.info(f"\tChannels:{self.conv_parameters.channels}")
        self.logger.info(f"\tOutputDir:{self.results_dir}")
