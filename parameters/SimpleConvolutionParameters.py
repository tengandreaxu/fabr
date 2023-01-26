from typing import Optional


class SimpleConvolutionParameters:
    def __init__(
        self,
        channels: list,
        global_average_pooling: bool,
        batch_norm: Optional[bool] = True,
    ):
        self.channels = channels
        self.global_average_pooling = global_average_pooling
        self.batch_norm = batch_norm
