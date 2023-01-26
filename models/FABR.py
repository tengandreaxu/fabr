import os
import time
import numpy as np
import logging
import torch
from typing import Optional
from rf.RandomFeaturesType import RandomFeaturesType
from utils.file_handlers import save_pickle
from utils.printing import print_header
from models.utils import get_block_sizes
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from models.RandomFeaturesPredictor import RandomFeaturesPredictor
from utils.smart_ridge_evaluation import (
    get_accuracy_multiclass_dataframe,
)
from utils.multiclassification import get_predictions

logging.basicConfig(level=logging.INFO)


class FABR:
    def __init__(
        self,
        rf_type: RandomFeaturesType,
        shrinkage_list: list,
        small_subset_size: int,
        max_multiplier: int = 0,
        debug: bool = False,
        seed: int = 0,
        shift_random_seed: bool = False,
        voc_grid=None,
        features=None,
        block_sizes=None,
        number_random_features=None,
        reduce_voc_curve: int = 0,
        produce_voc_curve: bool = False,
        produce_betas: Optional[bool] = True,
        save_q_vector: Optional[bool] = False,
        just_use_original_features: Optional[bool] = False,
        use_numpy: Optional[bool] = False,
    ):
        self.name = f"FABR"
        self.debug = debug
        self.rf_type = rf_type
        self.small_subset_size = small_subset_size
        self.seed = seed
        self.reduce_voc_curve = reduce_voc_curve
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.produce_voc_curve = produce_voc_curve
        self.shrinkage_list = shrinkage_list
        self.produce_betas = produce_betas
        self.save_q_vector = save_q_vector
        self.max_multiplier = max_multiplier
        self.just_use_original_features = just_use_original_features
        self.logger = logging.getLogger("FABR")
        # For prediction
        self.number_labels = 0
        self.output = {}
        self.voc_curve = None
        self.block_sizes = []
        self.shift_random_seed = shift_random_seed
        self.use_numpy = use_numpy
        self.betas = None
        self.betas_name = f"betas_{seed}.pickle"
        if self.produce_voc_curve:
            self.betas_name = f"betas_voc_curve_{seed}.pickle"

        self.block_sizes = block_sizes
        self.number_random_features = number_random_features
        self.features = features
        self.voc_grid = voc_grid
        self.voc_curve_initialized = (
            self.block_sizes is not None
            and self.number_random_features is not None
            and self.features is not None
            and self.number_labels is not None
        )

    def set_seed(self, seed: int):
        # overwrites initialization
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.logger.info(f"Seed set to {seed}")

    def build_the_q_vector(self, psi_matrix: np.ndarray, number_random_features: int):
        """

        THE OUTPUT q_vector IS A NUMPY ARRAY FOR A REGRESSION PROBLEM
        (WHEN y_train HAVE ONE COLUMN)
        BUT IF IT IS A LIST FOR A CLASSIFICATION PROBLEM WHEN y_train HAVE MANY COLUMNS!
        IN THIS CASE, THE LENGTH OF THE LIST q_vector equals the number of label columns

        """

        sample_size = psi_matrix.shape[0]

        covariance = psi_matrix / sample_size

        # this is T \times T
        # signals.shape[0] is the number of observations
        # Bottleneck when high dimensional
        start = time.monotonic()
        from scipy import linalg

        if self.use_numpy:
            import numpy.linalg as linalg
        eigval, eigvec1 = linalg.eigh(covariance)  # np.linalg.eigh(covariance)
        end = time.monotonic()
        eig_val_time = end - start
        self.logger.info(f"Time to decompose psi_matrix: {eig_val_time:.2f}s")
        # now we filter away low eigenvalues.
        # Why is this legal?
        # So we are using (zI + X'X)^{-1} X'y
        # we have the polar decomposition X'=HV
        # (we do not use it in the analysis, so this is just for mathematical explanations)
        # Then, X'X= HVV'H=H^2.
        # So we are doing (zI+H^2)^{-1}H Vy
        # Then, the key observation is that if H has a kernel we can ignore its action on the kernel
        # Namely, H^2 = U D U' and (zI+H^2)^{-1}H = U (z+D)^{-1} D U'
        # and we see that whatever happens on where D=0 gets annihilated

        eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]
        logging.info(f"Selected Eigval: {len(eigval)}")
        # now eigvec1 is a bit smaller, T \times T1 for some T1<T

        # so here we could potentially do many columns in the y_train

        multiplied = (1 / eigval).reshape(-1, 1) * (
            eigvec1.T @ (covariance @ self.y_train)
        )
        # this vector is now T1 \times number_label_columns (if we have 1-hot encoding)

        # here it is subtle as the dimension of eigvec might be lower than that of beta !!!
        # but normalized has the right dimension !!
        if (
            len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1
        ):  # then we are doing multi-class

            normalized = [
                np.concatenate(
                    [
                        (1 / (eigval + z)).reshape(-1, 1)
                        * multiplied[:, i].reshape(-1, 1)
                        for z in self.shrinkage_list
                    ],
                    axis=1,
                )
                for i in range(multiplied.shape[1])
            ]

            q_vector = [eigvec1 @ normalized[i] for i in range(multiplied.shape[1])]
        else:

            # this is (T \times T1) \times (T1 \times len(shrinkage))
            # which should give T \times len(shrinkage)

            normalized = np.concatenate(
                [
                    (1 / (eigval + z)).reshape(-1, 1) * multiplied.reshape(-1, 1)
                    for z in self.shrinkage_list
                ],
                axis=1,
            )

            q_vector = [eigvec1 @ normalized]

        len_eigenval = len(eigval.tolist())
        if len_eigenval < number_random_features:
            # the true psi matrix is P \times P/ If P>T, then it will have zeros
            eigval = np.array(
                eigval.tolist() + [0] * (number_random_features - len_eigenval)
            )
        else:
            # otherwise the first number_random_features - len(psi_hat_eig.tolist())
            # eigenvalues are identically zero
            eigval = eigval[(number_random_features - len_eigenval) :]

        return q_vector, eigval

    def build_single_q_vector(
        self, psi_matrix: np.ndarray, number_random_features: int
    ):
        """
        When we don't need the voc_curve, we can be more memory efficient
        """

        sample_size = psi_matrix.shape[0]

        # N \times N
        psi_matrix /= sample_size

        start = time.monotonic()
        from scipy import linalg

        if self.use_numpy:
            import numpy.linalg as linalg

        eigval, eigvec1 = linalg.eigh(psi_matrix)
        end = time.monotonic()
        eig_val_time = end - start
        self.logger.info(f"Time to decompose psi_matrix: {eig_val_time:.2f}s")
        eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]
        logging.info(f"Selected Eigval: {len(eigval)}")

        multiplied = (1 / eigval).reshape(-1, 1) * (
            eigvec1.T @ (psi_matrix @ self.y_train)
        )
        if (
            len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1
        ):  # then we are doing multi-class

            normalized = [
                np.concatenate(
                    [
                        (1 / (eigval + z)).reshape(-1, 1)
                        * multiplied[:, i].reshape(-1, 1)
                        for z in self.shrinkage_list
                    ],
                    axis=1,
                )
                for i in range(multiplied.shape[1])
            ]

            q_vector = [eigvec1 @ normalized[i] for i in range(multiplied.shape[1])]
        else:

            # this is (T \times T1) \times (T1 \times len(shrinkage))
            # which should give T \times len(shrinkage)

            normalized = np.concatenate(
                [
                    (1 / (eigval + z)).reshape(-1, 1) * multiplied.reshape(-1, 1)
                    for z in self.shrinkage_list
                ],
                axis=1,
            )

            q_vector = [eigvec1 @ normalized]

        len_eigenval = len(eigval.tolist())
        if len_eigenval < number_random_features:
            # the true psi matrix is P \times P/ If P>T, then it will have zeros
            eigval = np.array(
                eigval.tolist() + [0] * (number_random_features - len_eigenval)
            )
        else:
            # otherwise the first number_random_features - len(psi_hat_eig.tolist())
            # eigenvalues are identically zero
            eigval = eigval[(number_random_features - len_eigenval) :]

        return q_vector, eigval

    def compute_psi_matrix_and_q_vectors_for_voc_grid(
        self,
        sample_size: int,
    ):
        """"""
        np.random.seed(self.seed)

        # 27Gb of zeros if sample_size=60k
        psi_matrix = np.zeros([sample_size, sample_size])

        if self.produce_voc_curve:
            psi_eigenvalues_for_expanding_complexity = dict()
            q_vectors_for_expanding_complexity = dict()
            sigma_hat_eigenvalues_for_expanding_complexity = dict()

        random_features_all = []
        k = 0
        total_q_vectors = 0
        print_header("Computing Psi Matrix and Q vector")
        for block in range(len(self.block_sizes) - 1):
            if block % 10 == 0:
                logging.info(f"Block {block}/{len(self.block_sizes)}")
            k += 1
            # now we loop through blocks of features
            number_features_in_subset = (
                self.block_sizes[block + 1] - self.block_sizes[block]
            )

            if self.just_use_original_features:
                random_features = self.x_train[
                    :, self.block_sizes[block] : self.block_sizes[block + 1]
                ]
            else:
                if self.shift_random_seed:
                    self.logger.info(
                        f"Geneating Random Features with Seed: {k+self.seed}"
                    )
                start = time.monotonic()
                random_features = RandomFeaturesGenerator.generate_random_features(
                    type=self.rf_type,
                    number_features_in_subset=number_features_in_subset,
                    features=self.x_train,
                    increment_seed=k if not self.shift_random_seed else (k + self.seed),
                )
                end = time.monotonic()
                self.logger.info(f"Generate random features time: {end-start:.2f}s")

            if self.debug:
                random_features_all.append(random_features)

            # this is the main bottleneck for big matrix
            psi_matrix += random_features @ random_features.T

            if self.produce_voc_curve and (
                self.block_sizes[block + 1] in self.voc_grid
            ):
                self.logger.info(
                    f"Total Q Vectors:\t{total_q_vectors + 1}/{len(self.voc_grid)-1}"
                )
                # so now we are running the regression on the intermediate
                # result with a subset of random features
                start = time.monotonic()
                q_vector, psi_hat_eig = self.build_the_q_vector(
                    psi_matrix,
                    number_random_features=self.block_sizes[block + 1],
                )

                q_vectors_for_expanding_complexity.update(
                    {self.block_sizes[block + 1]: q_vector}
                )

                psi_eigenvalues_for_expanding_complexity.update(
                    {self.block_sizes[block + 1]: psi_hat_eig}
                )
                end = time.monotonic()
                vector_time = end - start
                self.logger.info(f"Build The q vector: {vector_time:.3f}s")
                total_q_vectors += 1

        # Always False
        if self.debug:
            random_features = np.concatenate(random_features_all, axis=1)
            # Covariance matrix
            true_psi_matr = random_features.T @ random_features
        else:
            true_psi_matr = None
            random_features = None
        if self.produce_voc_curve:
            voc_curve = {
                "psi_eig": psi_eigenvalues_for_expanding_complexity,
                "q_vectors": q_vectors_for_expanding_complexity,
                "sigma_eig": sigma_hat_eigenvalues_for_expanding_complexity,
            }
        else:
            voc_curve = dict()
        return psi_matrix, true_psi_matr, random_features, voc_curve

    def _compute_voc_curve(self):
        if self.max_multiplier > 0:
            max_features = self.x_train.shape[0] * self.max_multiplier
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
                self.x_train.shape[0] * self.max_multiplier,
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

        self.number_random_features = self.features[-1]

        self.block_sizes = get_block_sizes(
            self.number_random_features, self.small_subset_size, self.voc_grid
        )

        self.logger.info(self.voc_grid)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Parameters
        ----------
        voc_grid: grid for producing VOC curve. Must be multiples of small_subset_size
        produce_betas : If True, then we also output the giant beta vector.
        It could be huge (size = number_random_features, which could be a million or so)
        produce_voc_curve : If True, then we actually output predictions for a giant grid of numbers of random features
        (with a step size of roughly number_random_features / small_subset_size)

        X_test : the chunk of out-of-sample (test) data on which we produce OOS predictions
        X_train : in-sample raw signals from which random features are constructed
        y_train : in-sample returns to be predicted
        number_random_features : how many random features we want to produce. Could be a very large number
        small_subset_size : we split random features into sub-groups so that they fit in memory and
        running it becomes feasible even on a small machine
        seed : random seed. One should run this for a fixed seed, and then average predictions across seeds

        """
        sample_size = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.number_labels = self.y_train.shape[1]
        if not self.voc_curve_initialized:
            self._compute_voc_curve()
        logging.info(f"# Block: {len(self.block_sizes) - 1}")
        start_psi_matrix = time.monotonic()

        (
            psi_matrix,
            self.true_psi_matr,
            self.random_features,
            voc_curve,
        ) = self.compute_psi_matrix_and_q_vectors_for_voc_grid(
            sample_size=sample_size,
        )

        end_psi_matrix = time.monotonic()
        logging.info(
            f"Psi time: {(end_psi_matrix - start_psi_matrix):.3f}s \t RF: {self.number_random_features} \t P': {self.small_subset_size}"
        )

        if not self.produce_voc_curve:
            # q_vector \in R^{T\times len(shrinkage_list)}
            # but psi_hat_eig have lots of missing zeros. We should add them
            q_vector, psi_hat_eig = self.build_single_q_vector(
                psi_matrix, self.number_random_features
            )
            voc_curve = {}
            voc_curve["psi_eig"] = {self.number_random_features: psi_hat_eig}
            voc_curve["q_vectors"] = {self.number_random_features: q_vector}

        self.voc_curve = voc_curve

        if self.save_q_vector:
            save_pickle(q_vector, "test.pickle")

    def predict(self, x_test: np.ndarray):
        if not self.voc_curve:
            raise Exception("You should fit the model first")

        (
            betas,
            predictions,
            future_random_features_all,
            beta_norms,
            realized_in_sample_mean_predictions,
        ) = RandomFeaturesPredictor.compute_betas_and_predictions(
            seed=self.seed,
            shrinkage_list=self.shrinkage_list,
            test_and_train={"test": x_test, "train": self.x_train},
            block_sizes=self.block_sizes,
            voc_curve=self.voc_curve,
            test=self.debug,
            produce_betas=self.produce_betas,
            shift_random_seed=self.shift_random_seed,
            rf_type=self.rf_type,
            number_labels=self.number_labels,
            just_use_original_features=self.just_use_original_features,
        )
        self.x_test = x_test

        self.betas = betas
        self.predictions = predictions

        if self.debug:
            timing_ridge_start = time.monotonic()
            future_random_features_all = np.concatenate(
                future_random_features_all, axis=1
            )
            beta_true = (
                np.concatenate(
                    [
                        np.linalg.inv(
                            z * np.eye(self.number_random_features)
                            + self.true_psi_matr
                            / self.x_train.shape[0]
                            / self.number_random_features
                        )
                        @ (
                            self.random_features.T
                            / np.sqrt(self.number_random_features)
                        )
                        @ self.y_train
                        for z in self.shrinkage_list
                    ],
                    axis=1,
                )
                / self.x_train.shape[0]
            )

            future_predictions_true = (
                future_random_features_all
                @ beta_true
                / np.sqrt(self.number_random_features)
            )

            timing_ridge_end = time.monotonic()
            ridge_time = timing_ridge_end - timing_ridge_start

            self.output["beta_true"] = beta_true
            self.output["random_features"] = self.random_features
            self.output["future_random_features_all"] = future_random_features_all
            self.output["ridge_time"] = ridge_time
            self.output["ridge_predictions"] = future_predictions_true

    def compute_accuracy(self, y_test: np.ndarray):
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

    def get_betas(self, x_train: np.ndarray):
        if not self.voc_curve:
            raise Exception("You should fit the model first")

        betas = RandomFeaturesPredictor.get_betas(
            seed=self.seed,
            test_and_train={"train": x_train},
            block_sizes=self.block_sizes,
            voc_curve=self.voc_curve,
            rf_type=self.rf_type,
            number_labels=self.number_labels,
            just_use_original_features=self.just_use_original_features,
        )

        return betas

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
