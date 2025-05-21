import numpy as np

from changepoynt.simulation import base


class ConstantOffset(base.BaseTrend):
    offset = base.Parameter((float, int), limit=(-np.inf, np.inf), tolerance=0.1)

    def render(self) -> np.ndarray:
        return np.ones((self.length,))*self.offset


class LinearTrend(base.BaseTrend):
    offset = base.Parameter((float, int), limit=(-np.inf, np.inf), tolerance=0.1)
    slope = base.Parameter((float, int), limit=(-np.inf, 0, np.inf), tolerance=0.1)
    attachment_point = base.Parameter((float, int), limit=(-np.inf, np.inf), tolerance=0.1, modifiable=False, derived=True, use_random=False)

    def compute_attachment_point(self):
        # compute where the current slope ends
        return self.offset + self.slope * (self.shape[0] - 1)

    def render(self) -> np.ndarray:
        return self.offset + self.slope * np.linspace(0, self.shape[0]-1, self.shape[0])

    def __eq__(self, other):
        """
        We have to overwrite this function has we do not want to compare offset to offset but endpoint of ourselves
        to the other starting point (offset).

        :param other: the other object to compare against
        :return: bool
        """
        # if the other one is also a trend, we have to check whether they would attach to each other
        # without a problem
        if isinstance(other, LinearTrend):

            # check whether the end points and current points are completely different
            attachment_point_equal = abs(other.offset-self.attachment_point) <= self.parameter_info["attachment_point"]["tolerance"]

            # check whether the slopes are different
            slope_equal = abs(self.slope - other.slope) <= self.parameter_info["slope"]["tolerance"]
            return attachment_point_equal and slope_equal
        else:
            return False


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

    trend6 = LinearTrend(length=500, offset=499.0, slope=0.1) # fails