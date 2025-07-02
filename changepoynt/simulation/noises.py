import typing

import numpy as np

from changepoynt.simulation import base
import changepoynt.simulation.randomizers as rds


class NoNoise(base.BaseNoise):
    """
    This class is the default if no noise is specified. It adds no noise to the signal.
    """
    def render(self, *args, **kwargs) -> np.ndarray:
        return np.zeros((self.length,))


class GaussianNoise(base.BaseNoise):
    std = base.Parameter((float, int), limit=(0, np.inf), tolerance=0.05, default_parameter_distribution=rds.NoDistribution(0.05))
    seed = base.Parameter((int,), limit=(0, np.inf), default_parameter_distribution=rds.DiscreteUniformDistribution(0, 10000))

    def render(self) -> np.ndarray:
        random_generator = np.random.default_rng(self.seed)
        return random_generator.normal(0.0, self.std, self.length)


if __name__ == "__main__":
    n1 = GaussianNoise(100, std=10, seed=4)
    n2 = GaussianNoise(100, std=1000, seed=4)
    n3 = GaussianNoise(100, std=1000, seed=3)
    print(n1 == n2) # false
    print(n1 == n3) # false
    print(n1 != n2) # true
    print(n1 != n3) # true
