import numpy as np

import changepoynt.simulation.base as simbase
import changepoynt.simulation.randomizers as rds

class SlowSine(simbase.BaseTrend):
    amplitude = simbase.Parameter(float, 1.0, default_parameter_distribution=rds.ContinuousGaussianDistribution(10.0))

    def render(self) -> np.ndarray:
        return np.sin(np.linspace(start=0.0, stop=np.pi, num=self.length))*self.amplitude

"""
class GaussTrend(simbase.BaseTrend):
    amplitude = simbase.Parameter(float, 1.0, (-10.0, 10.0), default_parameter_distribution=rds.ContinuousGaussianDistribution(2.0, minimum=10.0, maximum=10.0))
    std = simbase.Parameter(float, 1.0, limit=(-np.inf, 0.0, np.inf), default_parameter_distribution=rds.ContinuousUniformDistribution(-10.0, 10.0))
    cut = simbase.Parameter(int, default_value=5, limit=(1, 10000), default_parameter_distribution=rds.DiscreteUniformDistribution(1, 10))
    def render(self) -> np.ndarray:
        x = np.linspace(-self.cut, self.cut, self.length)
        gausscurve = 1.0 / (np.sqrt(2.0 * np.pi) * self.std**2) * np.exp(-np.power((x - 0.0) / self.std, 2.0) / 2)
        return gausscurve
"""