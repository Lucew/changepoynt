# TODO: specify noise (gaussian noise)
# TODO: take care the trends are always different to NoNoise(offset of zero)
import numpy as np

import base


class GaussianNoise(base.BaseNoise):

    def __init__(self, mean: float, std: float, shape: tuple | base.Signal, random_seed: int = None):

        self.mean = mean
        self.std = std
        self.shape_tuple = base.Signal.translate_shape(shape)
        self.seed = np.random.RandomState(random_seed)

    @property
    def shape(self) -> tuple[int,]:
        return self.shape_tuple

    def render(self) -> np.ndarray:
        return self.seed.normal(self.mean, self.std, self.shape_tuple)

