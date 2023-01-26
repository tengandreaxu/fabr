import time
import numpy as np
import logging
from typing import Tuple
from rf.RandomFeaturesType import RandomFeaturesType
from utils.printing import print_header
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator

logging.basicConfig(level=logging.INFO)


class RandomFeaturesPredictor:
    def __init__(self):
        pass

    @staticmethod
    def update_predictions(
        predictions: dict,
        t_or_t: str,
        random_features: dict,
        beta_chunks: dict,
        normalize_p: bool,
        voc_curve: dict,
        block_sizes: list,
        block: int,
        number_labels: int,
    ):
        """

        predictions are very simple: just beta * random_features

        Parameters
        ----------
        predictions :
        t_or_t : 'test' or 'train'

        Returns
        -------

        """
        predictions[t_or_t].update(
            {
                (key, i): predictions[t_or_t][key, i]  # \hat{y} +=
                + random_features[t_or_t]  # S
                @ beta_chunks[key, i]  # \beta
                / (np.sqrt(key) if normalize_p else 1)
                for key in voc_curve["q_vectors"]
                if key >= block_sizes[block + 1]
                for i in range(number_labels)
            }
        )

    @staticmethod
    def initialize_predictions(
        data: np.ndarray, shrinkage_list: list, number_labels: int, voc_curve: dict
    ):

        # (complexity, label)
        predictions = {
            (key, i): np.zeros([data.shape[0], len(shrinkage_list)])
            for key in voc_curve["q_vectors"].keys()
            for i in range(number_labels)
        }
        return predictions

    @staticmethod
    def get_betas(
        seed: int,
        test_and_train: dict,
        block_sizes: list,
        voc_curve: dict,
        rf_type: RandomFeaturesType,
        number_labels: int,
        shift_random_seed: bool = False,
        normalize_p: bool = False,
        just_use_original_features: bool = False,
    ) -> dict:
        """

        :param seed:
        :param shrinkage_list:
        :param test_and_train:
        :param block_sizes:
        :param voc_curve:
        :param produce_betas:
        :param test:
        :param rf_type:
        :param number_labels:
        :param normalize_p:
        :return:
        """
        # here it is very important that we re-build the same seeds !!!!
        np.random.seed(seed)

        betas = {
            (key, i): [] for key in voc_curve["q_vectors"] for i in range(number_labels)
        }

        k = 0

        blocksss = range(len(block_sizes) - 1)
        for block in blocksss:
            print_header(str(block))
            start_block_time = time.monotonic()
            k += 1

            start = time.monotonic()

            number_features_in_subset = block_sizes[block + 1] - block_sizes[block]
            random_features = {
                key: RandomFeaturesGenerator.generate_random_features(
                    type=rf_type,
                    number_features_in_subset=number_features_in_subset,
                    features=test_and_train[key],
                    increment_seed=k if not shift_random_seed else (k + seed),
                )
                if not just_use_original_features
                else test_and_train[key][:, block_sizes[block] : block_sizes[block + 1]]
                for key in test_and_train
            }

            end = time.monotonic()
            creation_time = end - start
            logging.info(f"Predictions RF creation time: {creation_time:.3f}s")

            start = time.monotonic()
            beta_chunks = {
                (key, i): (
                    random_features["train"].T
                    @ voc_curve["q_vectors"][key][i]
                    / (
                        test_and_train["train"].shape[
                            0
                        ]  # here the betas are finally divided by N
                        * np.sqrt(key if normalize_p else 1)
                    )
                )
                for key in voc_curve["q_vectors"]
                if key >= block_sizes[block + 1]
                for i in range(number_labels)
            }

            end = time.monotonic()
            chunks_time = end - start
            logging.info(f"Beta Chunks time: {chunks_time:.3f}s")

            betas.update(
                {
                    (key, i): betas[key, i] + [beta_chunks[key, i]]
                    for key in voc_curve["q_vectors"]
                    if key >= block_sizes[block + 1]
                    for i in range(number_labels)
                }
            )

            end_block_time = time.monotonic()
            block_time = end_block_time - start_block_time
            logging.info(f"Total pred. block time: {block_time:.3f}s")

        betas = {
            (key, i): np.concatenate(betas[key, i], axis=0)
            for key in voc_curve["q_vectors"]
            for i in range(number_labels)
        }

        return betas

    @staticmethod
    def compute_betas_and_predictions(
        seed: int,
        shrinkage_list: list,
        test_and_train: dict,
        block_sizes: list,
        voc_curve: dict,
        produce_betas: bool,
        test: bool,
        rf_type: RandomFeaturesType,
        number_labels: int,
        normalize_p: bool = False,
        y_train: np.ndarray = None,
        shift_random_seed: bool = False,
        just_use_original_features: bool = False,
    ) -> Tuple[dict, dict, list, dict, dict]:
        """

        :param seed:
        :param shrinkage_list:
        :param test_and_train:
        :param block_sizes:
        :param voc_curve:
        :param produce_betas:
        :param test:
        :param rf_type:
        :param number_labels:
        :param normalize_p:
        :return:
        """
        # here it is very important that we re-build the same seeds !!!!
        np.random.seed(seed)

        logging.info(f"Number of Labels: {number_labels}")
        # I am afraid to single out the next loop into a function so that the seed is not lost
        # first we initialize the output with empty lists and zeros
        betas = {
            (key, i): [] for key in voc_curve["q_vectors"] for i in range(number_labels)
        }

        # this quantities will be needed in RMT analysis.
        # while we of course overfit, but this is useful information for extracting the degree of overfit

        realized_in_sample_mean_predictions = {
            (key, i): np.zeros([1, len(shrinkage_list)])
            for key in voc_curve["q_vectors"].keys()
            for i in range(number_labels)
        }

        beta_norms = {
            (key, i): np.zeros([1, len(shrinkage_list)])
            for key in voc_curve["q_vectors"].keys()
            for i in range(number_labels)
        }

        predictions = {
            key: RandomFeaturesPredictor.initialize_predictions(
                test_and_train[key], shrinkage_list, number_labels, voc_curve
            )
            for key in ["test"]
        }

        future_random_features_all = list()
        k = 0

        blocksss = range(len(block_sizes) - 1)
        for block in blocksss:
            print_header(str(block))
            start_block_time = time.monotonic()
            k += 1
            # logging.info(f"Predictions for block: {k}/{len(block_sizes) - 1}")
            # now we loop through blocks of features

            start = time.monotonic()

            number_features_in_subset = block_sizes[block + 1] - block_sizes[block]

            random_features = {
                key: RandomFeaturesGenerator.generate_random_features(
                    type=rf_type,
                    number_features_in_subset=number_features_in_subset,
                    features=test_and_train[key],
                    increment_seed=k if not shift_random_seed else (k + seed),
                )
                if not just_use_original_features
                else test_and_train[key][:, block_sizes[block] : block_sizes[block + 1]]
                for key in test_and_train
            }

            end = time.monotonic()
            creation_time = end - start
            logging.info(f"Predictions RF creation time: {creation_time:.3f}s")
            if test:
                future_random_features_all.append(random_features["test"])

            # q_vector is T \times len(shrinkage_list)
            # random_features is T \times P1
            # hence beta_chunk \in \R^{P_1\times len(shrinkage_list)}
            # so the betas for the chunk will only matter for a model with hih enough complexity
            # hence the condition key >= block_sizes[block + 1]

            start = time.monotonic()

            beta_chunks = {
                (key, i): (
                    random_features["train"].T
                    @ voc_curve["q_vectors"][key][i]
                    / (
                        test_and_train["train"].shape[
                            0
                        ]  # here the betas are finally divided by N
                        * np.sqrt(key if normalize_p else 1)
                    )
                )
                for key in voc_curve["q_vectors"]
                if key >= block_sizes[block + 1]
                for i in range(number_labels)
            }

            end = time.monotonic()
            chunks_time = end - start
            logging.info(f"Beta Chunks time: {chunks_time:.3f}s")

            start = time.monotonic()

            RandomFeaturesPredictor.update_predictions(
                predictions,
                t_or_t="test",
                random_features=random_features,
                beta_chunks=beta_chunks,
                normalize_p=normalize_p,
                voc_curve=voc_curve,
                block_sizes=block_sizes,
                block=block,
                number_labels=number_labels,
            )

            end = time.monotonic()
            future_time = end - start
            logging.info(f"future predictions time: {future_time:.3f}s")
            # logging.info(f"In sample time: {in_sample_time:.3f}s")
            # same here: only stuff with high complexity,
            # if key >= block_sizes[block + 1], gets updated

            # so the amazing thing is that we do not need to actually store the betas.
            # we update predictions chunk-by-chunk and can forget them
            if produce_betas:
                betas.update(
                    {
                        (key, i): betas[key, i] + [beta_chunks[key, i]]
                        for key in voc_curve["q_vectors"]
                        if key >= block_sizes[block + 1]
                        for i in range(number_labels)
                    }
                )

            end_block_time = time.monotonic()
            block_time = end_block_time - start_block_time
            logging.info(f"Total pred. block time: {block_time:.3f}s")
        # here we divide by T, because the estimator of b_star_hat_in_sample
        # is designed to take stuff normalized by T
        if produce_betas:
            betas = {
                (key, i): np.concatenate(betas[key, i], axis=0)
                for key in voc_curve["q_vectors"]
                for i in range(number_labels)
            }

        return (
            betas,
            predictions,
            future_random_features_all,
            beta_norms,
            realized_in_sample_mean_predictions,
        )
