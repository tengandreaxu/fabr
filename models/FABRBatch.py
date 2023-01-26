import os
import torch
import numpy as np
from typing import Optional
from parameters import SimpleConvolutionParameters
from models.FABR import FABR
from models.SimpleConvolution import SimpleConvolution
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from rf.RandomFeaturesType import RandomFeaturesType
from models.utils import get_block_sizes
from utils.file_handlers import save_pickle, load_pickle
from utils.smart_ridge_evaluation import (
    get_accuracy_multiclass_dataframe,
)
from utils.multiclassification import get_predictions
import logging

logging.basicConfig(level=logging.INFO)


class FABRBatch:
    """Fast Annihilating Batch Regression"""

    def __init__(
        self,
        rf_type: RandomFeaturesType,
        shrinkage_list: list,
        small_subset_size: int,
        batch_size: int,
        max_multiplier: int,
        pred_batch_size: int = 0,
        debug: bool = False,
        seed: int = 0,
        convolution_parameters: SimpleConvolutionParameters = None,
        produce_voc_curve: bool = False,
        reduce_voc_curve: int = 0,
        shift_random_seed: bool = False,
        produce_betas: Optional[bool] = False,
        min_threshold: Optional[float] = 10 ** (-10),
        just_use_original_features: Optional[bool] = False,
    ):
        """FABR implementing the double stochastic gradient
        descent"""
        self.logger = logging.getLogger("FABRBatch")
        self.rf_type = rf_type
        self.shrinkage_list = shrinkage_list
        self.small_subset_size = small_subset_size
        self.batch_size = batch_size
        self.reduce_voc_curve = reduce_voc_curve
        self.debug = debug
        self.shift_random_seed = shift_random_seed
        self.seed = seed
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.produce_voc_curve = produce_voc_curve
        self.produce_betas = produce_betas
        self.min_threshold = min_threshold
        self.just_use_original_features = just_use_original_features
        self.name = "FABRBatch"
        self.convolution_parameters = convolution_parameters
        self.max_multiplier = max_multiplier
        self.pred_batch_size = pred_batch_size
        self.betas = None
        self.betas_name = f"betas_{seed}.pickle"
        if self.produce_voc_curve:
            self.betas_name = f"betas_voc_curve_{seed}.pickle"

    def _build_batches(self):
        self.batches = []

        index = 0
        while index < self.nbatches:

            start_index = index * self.batch_size
            end_index = (index + 1) * self.batch_size

            self.batches.append(
                (
                    self.x_train[start_index:end_index],
                    self.y_train[start_index:end_index],
                )
            )

            index += 1

        self.logger.info(f"\tNumber of batches:{self.nbatches}")
        self.logger.info(f"\tAverage batch size:{self.batches[0][0].shape}")

    def _compute_voc_curve(self):

        self.nbatches = self.x_train.shape[0] // self.batch_size
        if (self.x_train.shape[0] % self.batch_size) != 0:
            self.nbatches += 1
        if self.max_multiplier > 0:
            max_features = int(self.x_train.shape[0] * self.max_multiplier)
            self.features = np.arange(
                0,
                max_features,
                step=self.small_subset_size,
            )
            if max_features not in self.features:
                self.features = np.concatenate(
                    [self.features, np.array([max_features])]
                )
        else:
            self.features = np.array([0, self.x_train.shape[1]])

        self.voc_grid = self.features
        logging.info("Starting Ridge")

        if self.reduce_voc_curve > 0:

            self.voc_grid = np.arange(
                0,
                int(self.x_train.shape[0] * self.max_multiplier),
                step=self.reduce_voc_curve,
            )
            double_descent_zoom = np.linspace(
                0.1 * self.x_train.shape[0],
                self.x_train.shape[0],
                endpoint=False,
                num=3,
            )
            double_descent_zoom = double_descent_zoom.astype(int)

            double_descent_zoom = np.concatenate(
                [double_descent_zoom, self.voc_grid[:5]]
            )
            double_descent_zoom.sort()
            self.voc_grid = self.voc_grid[5::10]
            self.voc_grid = np.concatenate([double_descent_zoom, self.voc_grid])
            if self.features[-1] not in self.voc_grid:
                self.voc_grid = np.append(self.voc_grid, self.features[-1])
        if not isinstance(self.voc_grid, list):
            self.voc_grid = self.voc_grid.tolist()

        self.number_labels = self.y_train.shape[1]
        self.number_random_features = self.features[-1]
        self.block_sizes = get_block_sizes(
            self.number_random_features, self.small_subset_size, self.voc_grid
        )
        self.logger.info(self.voc_grid)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """ """
        self.x_train = x_train
        self.y_train = y_train
        self._compute_voc_curve()
        self._build_batches()

        i = 0
        for batch in self.batches:
            i += 1
            self.logger.info(f"\tFitting batch #: {i}")
            x_train, y_train = batch

            # Applies convolution if requested
            if self.convolution_parameters is not None:
                x_train = self.apply_convolution(
                    x_train=x_train, batch_number=i, train_test="train"
                )

            giant = FABR(
                rf_type=self.rf_type,
                shrinkage_list=self.shrinkage_list,
                max_multiplier=self.max_multiplier,
                voc_grid=self.voc_grid,
                number_random_features=self.number_random_features,
                features=self.features,
                block_sizes=self.block_sizes,
                small_subset_size=self.small_subset_size,
                produce_betas=self.produce_betas,
                produce_voc_curve=self.produce_voc_curve,
                shift_random_seed=self.shift_random_seed,
                just_use_original_features=self.just_use_original_features,
                seed=self.seed,
            )

            giant.fit(x_train=x_train, y_train=y_train)

            if self.betas is None:
                self.betas = giant.get_betas(x_train)

            else:
                betas_chunk = giant.get_betas(x_train)

                for key in betas_chunk.keys():
                    self.betas[key] += betas_chunk[key]

        # ****************
        # scaling back betas
        # ****************
        for key in self.betas.keys():
            self.betas[key] = self.betas[key] / self.nbatches

    def apply_convolution(
        self, x_train: np.ndarray, batch_number: int, train_test: str
    ) -> np.ndarray:
        folder = f"data/preprocessed/batch/mnist1m"
        os.makedirs(folder, exist_ok=True)
        unique_name = f"size_{self.batch_size}_batch_{batch_number}_norm_{self.convolution_parameters.batch_norm}_seed_{self.seed}_channels_{self.convolution_parameters.channels}_gpa_{self.convolution_parameters.global_average_pooling}"

        X_train_name = os.path.join(folder, f"{unique_name}_X_{train_test}.pickle")

        if os.path.exists(X_train_name):
            x_train = load_pickle(X_train_name)

        else:
            simple_convolution = SimpleConvolution(
                in_channels=x_train[0].shape[0],
                channels=self.convolution_parameters.channels,
                global_average_pooling=self.convolution_parameters.global_average_pooling,
                seed=self.seed,
            )
            with torch.no_grad():
                x_train = simple_convolution(x_train).detach().numpy()
            save_pickle(x_train, X_train_name)
        return x_train

    def predict(self, x_test: np.ndarray):
        assert self.betas is not None, "fit() the model first!"
        self.x_test = x_test
        if self.pred_batch_size > 0:
            index = 0
            batches = self.x_test.shape[0] // self.pred_batch_size
            self.predictions = {}
            while index < batches:

                start_index = index * self.pred_batch_size
                end_index = (index + 1) * self.pred_batch_size
                x_test_chunk = self.x_test[start_index:end_index]

                predictions_chunk = self._predict(x_test_chunk, batch_number=index)

                if len(self.predictions) == 0:
                    self.predictions = predictions_chunk

                else:

                    self.predictions["test"].update(
                        {
                            key: np.vstack(
                                [
                                    self.predictions["test"][key],
                                    predictions_chunk["test"][key],
                                ]
                            )
                            for key in self.predictions["test"].keys()
                        }
                    )
                index += 1

        else:
            self.predictions = self._predict(self.x_test, batch_number=0)

    def _predict(self, x_test: np.ndarray, batch_number: int) -> dict:
        k = 0

        if self.convolution_parameters is not None:
            x_test = self.apply_convolution(
                x_test, batch_number=batch_number, train_test="test"
            )

        predictions = {
            "test": {
                key: np.zeros([x_test.shape[0], len(self.shrinkage_list)])
                for key in self.betas.keys()
            }
        }
        if not self.just_use_original_features:
            for block in self.block_sizes:

                k += 1
                if k == len(self.block_sizes):
                    break
                number_features_in_subset = (
                    self.block_sizes[k] - self.block_sizes[k - 1]
                )

                # Generate random features with the test set
                S_k = RandomFeaturesGenerator.generate_random_features(
                    type=self.rf_type,
                    number_features_in_subset=number_features_in_subset,
                    features=x_test,
                    increment_seed=k if not self.shift_random_seed else (k + self.seed),
                )

                beta_chunks = {
                    key: self.betas[key][self.block_sizes[k - 1] : self.block_sizes[k]]
                    for key in self.betas
                }

                # \hat{y} += \beta_k S_k

                predictions["test"].update(
                    {
                        key: predictions["test"][key]  # \hat{y} +=
                        + S_k @ beta_chunks[key]  # S  # \beta
                        for key in self.betas
                        if key[0] >= self.block_sizes[k]
                    }
                )
        return predictions

    def compute_accuracy(self, y_test: np.ndarray):
        # We have one-against-all predictions, a matrix in \R ^ {test_size, number_labels}
        if self.produce_voc_curve:
            complex_ = np.delete(self.voc_grid, 0)
            complex_ = complex_.tolist()
        else:
            complex_ = [self.number_random_features]
        self.y_test = y_test
        self.predictions["future_predictions_multiclass"] = get_predictions(
            {"future_predictions": self.predictions},
            range(self.number_labels),
            self.shrinkage_list,
            complex_,
        )

        self.accuracies = get_accuracy_multiclass_dataframe(
            self.predictions,
            self.y_test.argmax(1).reshape(-1, 1),
            self.shrinkage_list,
            complex_,
        )

        self.accuracies.index = self.accuracies.index / self.x_train.shape[0]

    def load_model(self, folder: str):
        import pickle

        self.betas = pickle.load(open(os.path.join(folder, self.betas_name), "rb"))

    def save_betas(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, self.betas_name), "wb") as handle:
            import pickle

            pickle.dump(self.betas, handle)

    def save_predictions(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        save_pickle(
            self.predictions, os.path.join(folder, f"predictions_{self.seed}.pickle")
        )

    def save_y_test(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        save_pickle(self.y_test, os.path.join(folder, f"y_test_{self.seed}.pickle"))

    def save_accuracies(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.accuracies.to_csv(os.path.join(folder, f"accuracies_{self.seed}.csv"))

    def is_beta_already_computed(self, output_dir: str) -> bool:

        betas_file = os.path.join(output_dir, self.betas_name)
        return os.path.exists(betas_file)
