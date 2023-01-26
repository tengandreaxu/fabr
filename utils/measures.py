import os
import pandas as pd
import numpy as np


def mean_max_accuracies(runs: int, result_dir: str, max_complexity: float = None):
    max_accuracies = []
    for run in range(runs):
        df = pd.read_csv(os.path.join(result_dir, f"accuracies_{run}.csv"))
        if "Unnamed: 0" in df:
            df = df.rename(columns={"Unnamed: 0": "complexity"})
            df.complexity = df.complexity.astype(float)
            if max_complexity is not None:
                df = df[df.complexity <= max_complexity].copy()
            df.pop("complexity")

        max_accuracy = df.max()
        max_accuracies.append(max_accuracy.T)

    mean_max_accuracy = pd.concat(max_accuracies).reset_index()
    mean_max_accuracy.columns = ["shrinkage", "mean"]
    std_max_accuracy = mean_max_accuracy.groupby("shrinkage").std().reset_index()
    std_max_accuracy = std_max_accuracy.rename(columns={"mean": "std"})
    mean_max_accuracy = mean_max_accuracy.groupby("shrinkage").mean().reset_index()

    mean_max_accuracy.shrinkage = mean_max_accuracy.shrinkage.astype(float)
    std_max_accuracy.shrinkage = std_max_accuracy.shrinkage.astype(float)
    return mean_max_accuracy, std_max_accuracy


def accuracy_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """computes the accuracy given a matrix of predictions

    Args:

        y_test, columnwise label
        y_hat, matrix of predictions where each column is a model prediction
    """

    return (y_test == y_hat).sum(0) / y_test.shape[0]


def r2(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    err = ((y_test - y_hat) ** 2).sum()
    bench = ((y_test - y_test.mean()) ** 2).sum()
    return 1 - (err / bench)


def mse_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """y_hat is a matrix of predictions"""
    samples = y_test.shape[0]
    return (((y_test - y_hat) ** 2) / samples).sum(0)


def r2_matrix(y_test: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """y_hat is a matrix of predictions"""
    return 1 - ((y_test - y_hat) ** 2).sum(0) / ((y_test - y_test.mean()) ** 2).sum(0)


def columnwise_mse_regression(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    return (((y_test - y_hat) ** 2) / y_test.shape[0]).sum()


def columnwise_root_mse_regression(y_test: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sqrt(columnwise_mse_regression(y_test, y_hat))
