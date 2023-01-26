from dataclasses import dataclass


@dataclass
class RFCifar10Parameters:

    shrinkages = [
        1e-5,
        0.1,
        1,
        2,
        3,
        5,
        10,
        100,
        200,
        1000,
        3000,
        5000,
        6000,
        10000,
        50000,
        100000,
    ]
    max_multiplier = 15
    channels = [64, 256, 1024, 8192]
    normalize = True
