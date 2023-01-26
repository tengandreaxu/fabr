import pandas as pd
import numpy as np
from utils.measures import accuracy_matrix, r2_matrix, mse_matrix


def get_r2_dataframe(
    output: dict,
    y_test: np.ndarray,
    shrinkages: list,
    complexities: list,
) -> pd.DataFrame:
    """retuns the r2 datafame from the giant regression"""

    r_squared = {
        key: r2_matrix(y_test, output["future_predictions"]["test"][key])
        for key in output["future_predictions"]["test"]
    }
    r_squared = pd.DataFrame(r_squared).T
    r_squared.columns = shrinkages
    r_squared.index = complexities
    return r_squared


def get_accuracy_multiclass_dataframe(
    output: dict,
    y_test: np.ndarray,
    shrinkages: list,
    complexities: list,
) -> pd.DataFrame:
    """retuns the accuracy datafame from the giant multiclassification"""

    accuracy = {
        key: accuracy_matrix(
            y_test,
            output["future_predictions_multiclass"][key],
        )
        for key in output["future_predictions_multiclass"]
    }

    accuracy = pd.DataFrame(accuracy).T
    accuracy.columns = shrinkages
    accuracy.index = complexities
    return accuracy


def get_beta_norms_datafame(
    output: dict, shrinkages: list, complexities: list
) -> pd.DataFrame:
    """returns the beta norm datafame from the giant regression"""

    beta_norms = {
        key: np.linalg.norm(output["betas"][key], axis=0) for key in output["betas"]
    }
    beta_norms = pd.DataFrame(beta_norms).T
    beta_norms.columns = shrinkages
    beta_norms.index = complexities
    return beta_norms


def get_mse_dataframe(
    output: dict, y_test: np.ndarray, shrinkages: list, complexities: list
) -> pd.DataFrame:
    """returns the mse dataframe from the giant regression"""
    mse = {
        key: mse_matrix(y_test, output["future_predictions"]["test"][key])
        for key in output["future_predictions"]["test"]
    }
    mse = pd.DataFrame(mse).T
    mse.columns = shrinkages
    mse.index = complexities
    return mse
