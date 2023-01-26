import numpy as np
from typing import Tuple


def get_one_against_all_matrix(labels: np.ndarray) -> np.ndarray:
    """creates the one against all matrix"""
    unique_labels = np.unique(labels)

    one_against_all_labels = np.concatenate(
        [(labels == label).astype(int) for label in unique_labels], axis=1
    )
    return one_against_all_labels


def get_one_against_all_matrix_from_train_test(
    y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    train_size = y_train.shape[0]
    labels = np.concatenate([y_train, y_test])
    one_against_all_matrix = get_one_against_all_matrix(labels)
    one_against_all_matrix_demean = (
        one_against_all_matrix - one_against_all_matrix.mean(0).reshape(1, -1)
    )

    y_train = one_against_all_matrix_demean[:train_size]
    y_test = one_against_all_matrix_demean[train_size:]
    return y_train, y_test


def get_multiclass_argmax_predictions(
    output: dict,
    unique_labels: np.ndarray,
    number_random_features: int,
    shrinkage_list: np.ndarray,
) -> np.ndarray:
    """transforms the multiclass future_predictions into the classic predictions' matrix

    Args:
        output, the dict output from RandomFeatures
    """
    return np.concatenate(
        [
            np.concatenate(
                [
                    output["future_predictions"]["test"][number_random_features, i][
                        :, j
                    ].reshape(-1, 1)
                    for i in range(len(unique_labels))
                ],
                axis=1,
            )
            .argmax(1)
            .reshape(-1, 1)
            for j in range(len(shrinkage_list))
        ],
        axis=1,
    )


def get_predictions(
    output: dict, unique_labels: np.ndarray, shrinkage_list: list, voc_grid: list
):
    return {
        key: get_multiclass_argmax_predictions(
            output, unique_labels, key, shrinkage_list
        )
        for key in voc_grid
    }
