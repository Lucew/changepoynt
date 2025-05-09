# TODO: specify trends (linear, offset, maybe even signals)
# TODO: take care the trends are always different to NoTrends(offset of zero)

# TODO: rewrite everything to register parameters
import numpy as np

import base


class ConstantOffset(base.ConstantTrend):
    """
    Adding a constant to the signal.
    """
    def __init__(self, offset: float, shape: tuple[int] | base.Signal, tolerance: float = 0.05):
        super().__init__(offset, shape, tolerance)
        if offset == 0:
            raise ValueError("Constant offset cannot be 0. Use NoTrend class instead.")


class LinearTrend(base.BaseTrend):

    def __init__(self, offset: float, slope: float, shape: tuple[int] | base.Signal, tolerance: float = 0.1):
        super().__init__(tolerance)

        # save the variables
        self.shape_tuple = base.Signal.translate_shape(shape)
        self.offset = offset
        self.slope = slope

    def render(self) -> np.ndarray:
        return self.offset + self.slope * np.linspace(0, self.shape[0]-1, self.shape[0])

    @property
    def shape(self) -> tuple[int,]:
        return self.shape_tuple

    def __eq__(self, other):

        # if the other one is also a trend, we have to check whether they would attach to each other
        # without a problem
        if isinstance(other, LinearTrend):

            # compute where the current slope ends
            self_end_value = self.offset + self.slope * (other.shape[0]-1)

            # check whether the end points and current points are completely different
            attachment_point_equal = abs(other.offset-self_end_value) < self.tolerance

            # check whether the slopes are different
            slope_equal = abs(self.slope - other.slope) < self.tolerance
            return attachment_point_equal and slope_equal
        else:
            return False





