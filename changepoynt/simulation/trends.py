import numpy as np

from changepoynt.simulation import base


class ConstantOffset(base.BaseTrend):
    offset: base.Parameter(float, limit=(-np.inf, np.inf), tolerance=0.1)

    def render(self):
        return self.offset


class LinearTrend(base.BaseTrend):
    offset = base.Parameter(float, limit=(-np.inf, np.inf), tolerance=0.1)
    slope = base.Parameter(float, limit=(-np.inf, np.inf), tolerance=0.1)
    attachment_point = base.Parameter(float, limit=(-np.inf, np.inf), tolerance=0.1, modifiable=False)

    def render(self) -> np.ndarray:
        return self.offset + self.slope * np.linspace(0, self.shape[0]-1, self.shape[0])

    def __eq__(self, other):

        # if the other one is also a trend, we have to check whether they would attach to each other
        # without a problem
        if isinstance(other, LinearTrend):

            # compute where the current slope ends
            self_end_value = self.offset + self.slope * (other.shape[0]-1)

            # check whether the end points and current points are completely different
            attachment_point_equal = abs(other.offset-self_end_value) < self.attachment_point.tolerance

            # check whether the slopes are different
            slope_equal = abs(self.slope - other.slope) < self.slope.tolerance
            return attachment_point_equal and slope_equal
        else:
            return False





