import numpy as np
from typing import Optional
from rf.RandomFeaturesType import RandomFeaturesType
from rf.RandomNeurons import RandomNeurons
from models.FABRNu import FABRNu
from utils.multiclassification import get_predictions
from utils.printing import print_header
from utils.sampling import get_demean_labels
from utils.smart_ridge_evaluation import get_accuracy_multiclass_dataframe
from data.DatasetManager import DatasetsManager
from tests.test_fabr import (
    train_old_giant_regression_multiclass,
)


def train_linear_spectral_method_multiclass(
    X_train: np.ndarray,
    y_train_demeaned: np.ndarray,
    seed: int,
    X_test: np.ndarray,
    shrinkage_list: list,
    rf_type: RandomFeaturesType,
    small_subset_size: int,
    niu: int,
    y_test_labels: np.ndarray,
):

    spectral_regression = FABRNu(
        rf_type,
        shrinkage_list=shrinkage_list,
        small_subset_size=small_subset_size,
        debug=False,
        seed=seed,
        max_multiplier=10,
        produce_voc_curve=True,
        niu=niu,
    )

    spectral_regression.fit(
        x_train=X_train,
        y_train=y_train_demeaned,
    )

    spectral_regression.predict(X_test)
    spectral_regression.compute_accuracy(y_test_labels)
    print(spectral_regression.accuracies)


if __name__ == "__main__":
    sample_size = 100
    number_features = 10
    activation = "relu"
    dataset = "simulated_multiclass"
    dm = DatasetsManager()
    (x_train, y_train, x_test, y_test) = dm.get_synthetic_dataset_normal_dist(
        n_observations=sample_size,
        n_features=number_features,
        number_classes=2,
        split_number=int(sample_size / 2),
    )
    y_train_demeaned, y_test = get_demean_labels(y_train, y_test)

    shrinkage_list = [0.0001, 0.001, 0.1, 1, 10, 100, 1000]
    small_subset_size = 100

    seed = 0
    niu = 50
    rf_type = RandomNeurons(activation=activation)

    print_header("Old Giant Regression")
    train_old_giant_regression_multiclass(
        x_train,
        y_train_demeaned,
        shrinkage_list,
        small_subset_size,
        seed,
        x_test,
        rf_type,
        y_test,
    )

    print_header("New Spectral Method")
    train_linear_spectral_method_multiclass(
        X_train=x_train,
        y_train_demeaned=y_train_demeaned,
        shrinkage_list=shrinkage_list,
        small_subset_size=small_subset_size,
        seed=seed,
        X_test=x_test,
        rf_type=rf_type,
        y_test_labels=y_test,
        niu=niu,
    )
