import numpy as np
from models.FABR import FABR
from sklearn.linear_model import Ridge, RidgeClassifier
from data.DatasetManager import DatasetsManager
from rf.RandomNeurons import RandomNeurons

if __name__ == "__main__":
    """
    To show that we are solving

    ((X'X)/N) + zI)^(-1) X' (y/N)
    """
    dm = DatasetsManager()

    rf_type = RandomNeurons(activation="linear")

    n_observations = 10
    n_test = 2
    z = 1
    x_train, y_train, x_test, y_test = dm.load_synthetic_data(
        n_observations=(n_observations + n_test),
        n_features=10,
        split_size=2,
    )

    spectral = FABR(
        rf_type=rf_type,
        shrinkage_list=[z],
        small_subset_size=n_observations,
        just_use_original_features=True,
        debug=True,
    )

    spectral.fit(x_train, y_train)
    spectral.predict(x_test)
    spectral.compute_accuracy(y_test)
    true_betas = (
        np.linalg.inv(
            ((x_train.T @ x_train) / n_observations) + z * np.eye(n_observations)
        )
        @ (x_train.T)
        @ (y_train / n_observations)
    )

    print(f"True Betas: {true_betas}")
    print(f"FABR Betas: {spectral.betas}")
