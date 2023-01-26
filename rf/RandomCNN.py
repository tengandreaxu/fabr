from rf.RandomFeaturesType import RandomFeaturesType, RandomFeaturesNames


class RandomCNN(RandomFeaturesType):
    def __init__(self, channels: list, add_bias: bool):
        RandomFeaturesType.__init__(self, activation="", name=RandomFeaturesNames.cnn)
        self.channels = channels
        self.add_bias = add_bias
