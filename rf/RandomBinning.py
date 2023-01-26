from rf.RandomFeaturesType import RandomFeaturesType, RandomFeaturesNames


class RandomBinning(RandomFeaturesType):
    def __init__(self, activation: str):
        RandomFeaturesType.__init__(self, activation, RandomFeaturesNames.binning)
        self.name = RandomFeaturesNames.binning
        self.distribution = "standard_normal"
        self.distribution_parameters = []

    def to_string(self):

        if isinstance(self.activation, str):
            return f"{self.name}_{self.activation}_{self.distribution}_{self.distribution_parameters}"
        else:
            return f"{self.name}_{self.activation[0]}{self.activation[1]}_{self.distribution}_{self.distribution_parameters}"
