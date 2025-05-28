import numpy as np

from changepoynt.simulation import base


class NoNoise(base.BaseNoise):
    """
    This class is the default if no noise is specified. It adds no noise to the signal.
    """
    def render(self, *args, **kwargs) -> np.ndarray:
        return np.zeros((self.length,))


class GaussianNoise(base.BaseNoise):
    std = base.Parameter((float, int), limit=(-np.inf, 0, np.inf), tolerance=0.05)
    seed = base.Parameter(int, tolerance=0.1, modifiable=False, use_for_comparison=False)

    def render(self) -> np.ndarray:
        return self.seed.normal(0.0, self.std, self.length)


if __name__ == "__main__":
    n1 = GaussianNoise(100, mean=0.0, std=0.1, seed=42)
    n2 = GaussianNoise(100, mean=0.0, std=0.1, seed=1100)
    n3 = GaussianNoise(100, mean=20, std=0.1, seed=1100)
    print(n1 == n2) # true
    print(n1 == n3) # false
    print(n1 != n2) # false
    print(n1 != n3) # true
