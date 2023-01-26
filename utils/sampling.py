from typing import Tuple
import numpy as np
import random
from utils.multiclassification import get_one_against_all_matrix


def sample_balanced_trainset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_size: int,
    seed: int,
    labels: list,
    keep_torch: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """takes a balanced subsample from the trainset"""
    np.random.seed(seed)
    random.seed(seed)
    indexes = np.array([])
    for i in range(10):
        indices = np.where(y_train == i)[0]
        index_i = np.random.choice(indices, size=sample_size, replace=False)
        indexes = np.concatenate([indexes, index_i])

    X_train_out = X_train[indexes.astype(int)]
    y_train_out = y_train[indexes.astype(int)]
    # I need to shuffle again here
    X_train_y_train = [(x, y) for (x, y) in zip(X_train_out, y_train_out)]
    random.shuffle(X_train_y_train)

    if keep_torch:
        import torch

        X_train_out = torch.stack([k[0] for k in X_train_y_train])

    else:
        X_train_out = np.array([k[0] for k in X_train_y_train])
    y_train_out = np.array([k[1] for k in X_train_y_train])

    return X_train_out, y_train_out


def get_demean_labels(
    y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """returns the matrices given y_train and y_test for discrete classification"""
    labels = np.concatenate([y_train, y_test])
    one_against_all_matrix = get_one_against_all_matrix(labels)
    one_against_all_matrix_demean = (
        one_against_all_matrix - one_against_all_matrix.mean(0).reshape(1, -1)
    )

    train_size = y_train.shape[0]
    y_train = one_against_all_matrix_demean[:train_size]
    y_test = one_against_all_matrix_demean[train_size:]
    return y_train, y_test
