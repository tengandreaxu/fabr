import os
import time
import numpy as np
import logging
from copy import deepcopy
from typing import Optional, Tuple

from models.RandomFeaturesPredictor import RandomFeaturesPredictor
from rf.RandomFeaturesType import RandomFeaturesType
from utils.printing import print_header
from utils.file_handlers import save_pickle
from models.utils import get_block_sizes
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from utils.smart_ridge_evaluation import (
    get_accuracy_multiclass_dataframe,
)
from utils.multiclassification import get_predictions

logging.basicConfig(level=logging.INFO)


class FABRNu:
    def __init__(
        self,
        rf_type: RandomFeaturesType,
        shrinkage_list: list,
        small_subset_size: int,
        niu: int,
        debug: bool = False,
        seed: int = 0,
        reduce_voc_curve: int = 0,
        max_multiplier: int = 0,
        shift_random_seed: bool = False,
        produce_voc_curve: bool = False,
        produce_betas: Optional[bool] = False,
        min_threshold: Optional[float] = 10 ** (-10),
        just_use_original_features: Optional[bool] = False,
    ):
        self.debug = debug
        self.rf_type = rf_type
        self.niu = niu
        self.name = f"spectral_niu_{self.niu}"

        self.small_subset_size = small_subset_size
        self.seed = seed
        self.shift_random_seed = shift_random_seed
        self.produce_voc_curve = produce_voc_curve
        self.shrinkage_list = shrinkage_list
        self.produce_betas = produce_betas
        self.min_threshold = min_threshold
        self.reduce_voc_curve = reduce_voc_curve
        self.max_multiplier = max_multiplier
        self.logger = logging.getLogger("FABRNu")
        self.just_use_original_features = just_use_original_features
        self.Vs = dict()
        self.eigens = dict()
        self.voc_curve = dict()

        self.voc_curve["q_vectors"] = dict()

        self.betas_name = f"betas_{seed}.pickle"
        if self.produce_voc_curve:
            self.betas_name = f"betas_voc_curve_{seed}.pickle"

    def reset(self):
        self.number_labels = 0
        self.voc_curve = None
        self.block_sizes = []

    def _compute_voc_curve(self):
        """TODO a user may override the list of features
        and specify the total number of features"""

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

        # ***************
        # TODO we will keep the first 5 anyway in case of reducing voc_curve
        # ***************
        if self.reduce_voc_curve > 0:
            #     double_descent_zoom = self.features[:5]
            #     self.voc_grid = self.features[5 :: self.reduce_voc_curve]
            #     self.voc_grid = np.concatenate([double_descent_zoom, self.voc_grid])
            #     if self.features[-1] not in self.voc_grid:
            #         self.voc_grid = np.append(self.voc_grid, self.features[-1])
            # if not isinstance(self.voc_grid, list):
            #     self.voc_grid = self.voc_grid.tolist()
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

        self.number_labels = self.y_train.shape[1]
        self.number_random_features = self.features[-1]

        self.block_sizes = get_block_sizes(
            self.number_random_features, self.small_subset_size, self.voc_grid
        )

        self.logger.info(self.voc_grid)

    def populate_q_vectors_dictionary(
        self,
        block_sizes: list,
        block: int,
        q_vector: dict,
    ):
        if self.produce_voc_curve and (block_sizes[block + 1] in self.voc_grid):
            voc_curve_time = time.monotonic()
            if self.number_labels > 1:

                self.voc_curve["q_vectors"].update(
                    {block_sizes[block + 1]: deepcopy(q_vector)}
                )
            else:
                self.voc_curve["q_vectors"].update(
                    {block_sizes[block + 1]: [deepcopy(q_vector)]}
                )
            voc_curve_time_end = time.monotonic()
            voc_total_time = voc_curve_time_end - voc_curve_time
            logging.info(f"VOC curve update time: {voc_total_time:.3f}")

    def eigen_dec_with_prefilter(self, cov: np.ndarray):
        tilda_D, tilda_V = np.linalg.eigh(cov)

        # now we get rid of redundant eigenvalues.
        # This must be done before we proceed to matrix multiplication,
        # to save dimensions
        tilda_V = tilda_V[:, tilda_D > self.min_threshold]
        tilda_D = tilda_D[tilda_D > self.min_threshold]
        tilda_V = tilda_V[:, -min(self.niu, cov.shape[0]) :]
        tilda_D = tilda_D[-min(self.niu, cov.shape[0]) :]
        return tilda_D, tilda_V

    def eigen_decompositions_with_top_and_not_too_small_eigenvalues(
        self,
        features: np.ndarray,
    ):
        """
        So, our goal is to get the top eigenvectors and eigenvalues of AA', where A=features
        So we use smart algebra:
        A'Av = lambda v implies
        AA' (Av) = lambda (Av)
        and \|Av\|^2=v'A'Av=lambda. Thus,
        if we can eigen-decompose A'A"
        A'A = V D V'
        we have AA' = U (D)U'
        where U = [Av D^{-1/2}] are the eigenvalues of A'A
        Parameters
        ----------
        features :
        number_top_eigenvalues :
        gpu :

        Returns
        -------

        """
        p1 = features.shape[1]
        N = features.shape[0]

        if N > p1:
            features = features.T

        start = time.monotonic()

        cov = features @ features.T

        end = time.monotonic()
        s_s_time = end - start
        self.logger.info(f"S'_kS_k Time: {s_s_time:.2f}s")
        tilda_D, tilda_V = self.eigen_dec_with_prefilter(cov)

        # now, eigenvalues do not change, but eigenvectors, we need to be fixed:
        if N > p1:
            tilda_V = features.T @ (tilda_V * (tilda_D ** (-1 / 2)).reshape(1, -1))
        return tilda_V, tilda_D

    def produce_the_first_q_vector_for_zero_block(
        self,
        random_features,
        block,
    ):
        # \tilda(D), \tilda(V) = eigen(S_k'S_k)
        (
            V_0,
            tilda_D,
        ) = self.eigen_decompositions_with_top_and_not_too_small_eigenvalues(
            features=random_features,
        )

        logging.info(f"Selected Eigenvalues: {len(tilda_D)}")

        # dictionaries
        # here as well we are using that B@A=(diag(B)) * A for diagonal matrices
        if self.number_labels > 1:
            q_vector = [
                np.concatenate(
                    [
                        V_0
                        @ (
                            (1 / (tilda_D + z)).reshape(-1, 1)
                            * (V_0.T @ self.y_train[:, i].reshape(-1, 1))
                        )
                        for z in self.shrinkage_list
                    ],
                    axis=1,
                )
                for i in range(self.y_train.shape[1])
            ]
        else:
            start = time.monotonic()
            q_vector = np.concatenate(
                [
                    V_0 @ ((1 / (tilda_D + z)).reshape(-1, 1) * (V_0.T @ self.y_train))
                    for z in self.shrinkage_list
                ],
                axis=1,
            )
            end = time.monotonic()
            q_0_time = end - start
            self.logger.info(f"Q0 Vector {q_0_time:.2f}s")
        self.populate_q_vectors_dictionary(
            self.block_sizes,
            block,
            q_vector,
        )
        self.Vs[block] = V_0
        self.eigens[block] = tilda_D
        return q_vector

    def mathematical_q_vector_computation(self, V_k, diagonals, V_k_T_y_train):

        if self.number_labels > 1:
            # V_k has \nu columns, T rows; diagonal is \nu \times \nu
            q_vector = [
                np.concatenate(
                    [
                        V_k
                        @ (
                            np.diag(diagonal).reshape(-1, 1)
                            * V_k_T_y_train[:, i].reshape(-1, 1)
                        )
                        for diagonal in diagonals
                    ],
                    axis=1,
                )
                for i in range(V_k_T_y_train.shape[1])
            ]
        else:
            q_vector = np.concatenate(
                [
                    V_k
                    @ (np.diag(diagonal).reshape(-1, 1) * V_k_T_y_train.reshape(-1, 1))
                    for diagonal in diagonals
                ],
                axis=1,
            )

        return q_vector

    def produce_q_vector_for_nonzero_block(
        self,
        random_features,
        block,
    ):

        start_full_block = time.monotonic()
        # should be T x min(niu, P)
        previous_V = self.Vs[block - 1]
        previous_d = self.eigens[block - 1]
        # Define D_{k-1} = diag(d_{k-1})

        # **************************
        # Xi matrix computation and
        # decomposition takes < 2s
        # For N=42k, nu=600
        # **************************

        # \Xi_k = D_{k-1} + V'_{k-1} S_k S'_k V_{k-1}
        start = time.monotonic()
        tilda_V_k = previous_V.copy()
        end = time.monotonic()
        copying_time = end - start
        logging.info(f"previous_V copying time: {copying_time:.3f}s")
        start = time.monotonic()

        # now we build the S_k projected onto orthogonal complement of the
        # span of old eigenvectors

        # random_features are T \times P_1;
        # previous_V is T \times \nu
        # tilda_S_k is T \times P_1
        tilda_S_k = random_features - previous_V @ (previous_V.T @ random_features)

        end = time.monotonic()
        tilda_S_k_time = end - start
        logging.info(f"tilda_S_k time: {tilda_S_k_time:.3f}s")
        logging.info(f"tilda_S_k Shape should be T x P_1 : {tilda_S_k.shape}")
        start = time.monotonic()
        # \Gamma_k = \tilda(S_k)' \tilda(S_k)
        # \Gamma_k is P_1 \times P_1
        gamma_k = tilda_S_k.T @ tilda_S_k
        end = time.monotonic()
        gamma_k_time = end - start
        logging.info(f"gamma_k time: {gamma_k_time:.3f}")
        logging.info(f"gamma_k Shape should be P_1 x P_1: {gamma_k.shape}")
        start = time.monotonic()

        # W_k is T times P_1
        delta_k, W_k = self.eigen_dec_with_prefilter(
            gamma_k,
        )
        # \tilda(W)_k is T times P_1
        tilda_W_k = tilda_S_k @ (W_k * (1 / np.sqrt(delta_k)))
        # this are actually the orthogonalized columns of \tilde S

        # assert tilda_W_k.shape == (sample_size, p_1)
        end = time.monotonic()
        delta_k_tilda_W_k_time = end - start
        logging.info(f"delta_k, tilda_W_k time: {delta_k_tilda_W_k_time:.3f}s")

        # hat_V_k should be T \times (P1 + P1)
        hat_V_k = np.concatenate([tilda_V_k, tilda_W_k], axis=1)

        start = time.monotonic()
        # first we compute \hat V_k.T @ Psi_{k-1} @ \hat V_k\ =\ hat V_k.T @ V_{k-1} @ D @ V_{k-1}.T @ \hat V_k
        multiplied_v = hat_V_k.T @ previous_V
        hatv_times_psi_times_hatv_t = multiplied_v @ (
            previous_d.reshape(-1, 1) * multiplied_v.T
        )

        # new we compute the same for SS'
        multiplied_s = hat_V_k.T @ random_features
        hat_v_s_s_t_hat_v_t = multiplied_s @ multiplied_s.T
        psi_star = hatv_times_psi_times_hatv_t + hat_v_s_s_t_hat_v_t

        # now we have the (p_1+\nu)\times (p_1+\nu) matrix. And we diagonalize it
        lambda_k, V_k = self.eigen_dec_with_prefilter(cov=psi_star)
        V_k = hat_V_k @ V_k

        end = time.monotonic()
        V_k_time = end - start
        logging.info(f"V_k time: {V_k_time:.3f}s")
        self.eigens[block] = lambda_k

        self.Vs[block] = V_k

        if self.block_sizes[block - 1] not in self.voc_grid:
            del self.eigens[block - 1]
        del self.Vs[block - 1]

        logging.info(f"Selected Eigval: {len(lambda_k)}")

        # ******************
        # Q- Vector computation,
        # can take 10s for sample_size=42k and nu=600
        # ******************
        if self.produce_voc_curve and (self.block_sizes[block + 1] in self.voc_grid):

            start = time.monotonic()
            # matrix multiplication is associative, i.e. (AB)C = A(BC)
            diagonals = [np.diag(1 / (lambda_k + z)) for z in self.shrinkage_list]
            V_k_T_y_train = V_k.T @ self.y_train
            q_vector = self.mathematical_q_vector_computation(
                V_k, diagonals, V_k_T_y_train
            )

            end = time.monotonic()
            q_vector_time = end - start
            if self.debug:
                logging.info(f"Q-Vector time: {q_vector_time:.3f}s")

            self.populate_q_vectors_dictionary(
                self.block_sizes,
                block,
                q_vector,
            )
        end_full_block = time.monotonic()
        full_block_time = end_full_block - start_full_block

        logging.info(f"Full block time: {full_block_time:.3f}s")

    def generate_random_features(self, block: int):
        p_1 = self.block_sizes[block + 1] - self.block_sizes[block]
        # 1. Generate Random Features
        # Should be T x P_1
        start_rf = time.monotonic()
        if self.just_use_original_features:
            random_features = self.x_train[
                :, self.block_sizes[block] : self.block_sizes[block + 1]
            ]
        else:
            random_features = RandomFeaturesGenerator.generate_random_features(
                number_features_in_subset=p_1,
                features=self.x_train,
                increment_seed=(block + 1)
                if not self.shift_random_seed
                else (self.seed + block),
                type=self.rf_type,
            )
        # Divide them by T^1/2
        random_features = random_features / np.sqrt(self.sample_size)
        # this division ensures that we are doing S'S/N

        end_rf = time.monotonic()
        rf_time = end_rf - start_rf
        logging.info(f"RF creation time: {rf_time:.3f}")
        return random_features

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Paper Main Algorithm
        """
        self.x_train = x_train
        self.y_train = y_train
        self._compute_voc_curve()

        self.number_labels = self.y_train.shape[1]
        logging.info(f"# Block: {len(self.block_sizes) - 1}")
        logging.info(f"Voc Grid: {str(self.voc_grid)}")
        self.sample_size = self.x_train.shape[0]

        for block in range(len(self.block_sizes) - 1):
            # Chunk Size
            print_header(str(block))
            random_features = self.generate_random_features(block)

            if block == 0:
                self.produce_the_first_q_vector_for_zero_block(
                    random_features,
                    block,
                )
            else:
                self.produce_q_vector_for_nonzero_block(
                    random_features,
                    block,
                )

    def predict(
        self,
        X_test: np.ndarray,
    ):
        """

        :param X_test:
        :param X_train:
        :param y_train:
        :param eigenvalues: eigens dictionary, indexed by [block]
        :return:
        """
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
            test_and_train={"test": X_test, "train": self.x_train},
            block_sizes=self.block_sizes,
            voc_curve=self.voc_curve,
            produce_betas=self.produce_betas,
            test=self.debug,
            rf_type=self.rf_type,
            shift_random_seed=self.shift_random_seed,
            number_labels=self.number_labels,
            y_train=self.y_train,
            just_use_original_features=self.just_use_original_features,
        )

        self.betas = betas
        self.predictions = predictions

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

    def save_predictions(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        save_pickle(
            self.predictions, os.path.join(folder, f"predictions_{self.seed}.pickle")
        )

    def save_betas(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, self.betas_name), "wb") as handle:
            import pickle

            pickle.dump(self.betas, handle)

    def save_y_test(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        save_pickle(self.y_test, os.path.join(folder, f"y_test_{self.seed}.pickle"))

    def save_accuracies(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.accuracies.to_csv(os.path.join(folder, f"accuracies_{self.seed}.csv"))
