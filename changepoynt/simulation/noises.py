import numpy as np

from changepoynt.simulation import base


class NoNoise(base.BaseNoise):
    """
    This class is the default if no noise is specified. It adds no noise to the signal.
    """
    def render(self, *args, **kwargs) -> float:
        return 0


class GaussianNoise(base.BaseNoise):
    mean = base.Parameter(float, tolerance=0.05)
    std = base.Parameter(float, tolerance=0.05)
    seed = base.Parameter(int, tolerance=0.05, modifiable=False, use_for_comparison=False)

    def render(self) -> np.ndarray:
        return self.seed.normal(self.mean, self.std, self.length)
