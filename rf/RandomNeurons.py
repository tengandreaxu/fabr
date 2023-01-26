import numpy as np
from typing import Optional
from rf.RandomFeaturesType import RandomFeaturesType, RandomFeaturesNames


class RandomNeurons(RandomFeaturesType):
    def __init__(
        self,
        activation: str,
        gamma: Optional[list] = [1.0, 1.0],
    ):

        RandomFeaturesType.__init__(
            self,
            activation,
            RandomFeaturesNames.neurons,
        )
        self.distribution = "gaussian_mixture"
        self.distribution_parameters = gamma
        self.bias_distribution = None
        self.bias_distribution_parameters = [-np.pi, np.pi]

    def to_string(self):
        if self.bias_distribution is None:
            if isinstance(self.activation, str):
                return f"{self.name}_{self.activation}_{self.distribution}_{self.distribution_parameters}"
            else:
                return f"{self.name}_{self.activation[0]}{self.activation[1]}_{self.distribution}_{self.distribution_parameters}"
        else:
            raise Exception("Not Implemented")
