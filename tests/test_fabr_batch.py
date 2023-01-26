import numpy as np
from rf.RandomFeaturesType import RandomFeaturesType
from rf.RandomNeurons import RandomNeurons
from utils.sampling import get_demean_labels

from models.FABRBatch import FABRBatch
from data.DatasetManager import DatasetsManager


def train_fabr_batch(
    X_train: np.ndarray,
    y_train_demeaned: np.ndarray,
    shrinkage_list: list,
    small_subset_size: int,
    seed: int,
    X_test: np.ndarray,
    rf_type: RandomFeaturesType,
    y_test_labels: np.ndarray,
):

    giant_regression = FABRBatch(
        rf_type,
        shrinkage_list=shrinkage_list,
        small_subset_size=small_subset_size,
        batch_size=50,
        debug=False,
        seed=seed,
        max_multiplier=10,
        produce_voc_curve=True,
        pred_batch_size=25,
    )

    giant_regression.fit(
        x_train=X_train,
        y_train=y_train_demeaned,
    )
    giant_regression.predict(X_test)

    predicted_labels_method = "max_prediction"

    if predicted_labels_method == "max_prediction":

        giant_regression.compute_accuracy(y_test_labels)

        print(giant_regression.accuracies)


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
    train_fabr_batch(
        x_train,
        y_train_demeaned,
        shrinkage_list,
        small_subset_size,
        seed,
        x_test,
        RandomNeurons(activation=activation),
        y_test,
    )
