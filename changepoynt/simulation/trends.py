import numpy as np

from changepoynt.simulation import base
import changepoynt.simulation.randomizers as rds


class ConstantOffset(base.BaseTrend):
    offset = base.Parameter((float, int), limit=(-np.inf, np.inf), tolerance=0.1, default_parameter_distribution=rds.ContinuousConditionalGaussianDistribution(4))

    def render(self) -> np.ndarray:
        return np.ones((self.length,))*self.offset


class LinearTrend(ConstantOffset):
    slope = base.Parameter((float, int), limit=(-np.inf, 0, np.inf), tolerance=0.01, default_parameter_distribution=rds.ContinuousConditionalGaussianDistribution(.01, default_mean=0.0))

    def render(self) -> np.ndarray:
        return self.offset + self.slope * np.linspace(0, self.shape[0]-1, self.shape[0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # make a trend and attach a following one
    trend1 = LinearTrend(length=500, offset=0.0, slope=1.0)
    print(trend1.render()[-1])

    # make another trend
    trend2 = LinearTrend(length=500, offset=499.0, slope=1.09)
    print(trend1 == trend2) # should be true
    trend3 = LinearTrend(length=500, offset=499.0, slope=1.1)
    print(trend1 == trend3) # should be false
    trend4 = ConstantOffset(length=500, offset=499.0)
    print(trend1 == trend4) # should be false
    trend5 = ConstantOffset(length=500, offset=499.0)
    print(trend4 == trend5) # should be true

    # trend6 = LinearTrend(length=500, offset=499.0, slope=0.1) # fails