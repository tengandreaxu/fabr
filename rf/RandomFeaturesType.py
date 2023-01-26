class RandomFeaturesNames:
    binning = "binning"
    neurons = "neurons"
    cnn = "cnn"


class RandomFeaturesType:
    def __init__(
        self,
        activation: str,
        name: str,
):
        self.name = name
        self.activation = activation
