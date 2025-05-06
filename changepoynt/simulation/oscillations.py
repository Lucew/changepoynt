import numpy as np
import scipy.signal as spsig
import scipy.special as spspec

import base

# TODO: Scipy diric function. parameters: amplitude, periodicity. equality: amplitude, periodicity
# TODO: Sine. parameters: frequency, amplitude, phase. equality: amplitude, frequency, phase at change
# TODO: line. parameters: None. We will use the trend for further modifications
# TODO: rectangle pulse
# TODO: saw-tooth pulse
# TODO: ecg? but this introduces other dependencies
# TODO: signal from csv (equality, or as trend?)
# TODO: random walk
# TODO: check change point papers for what they simulate


class SineFunction(base.BaseOscillation):

    def __init__(self, length: int , periods: int = 10, amplitude: float = 1.0):

        # save the variables
        self.length = length
        self.amplitude = amplitude
        self.periods = periods

        # check that each period contains at least 5 samples
        if self.length < self.periods*5:
            raise ValueError("We require at least 5 samples per period")


    @property
    def shape(self) -> tuple[int,]:
        return (self.length,)

    def render(self):
        return self.amplitude*np.sin(np.linspace(0, self.periods * np.pi*2, self.length))

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about diric that might be similar?
        # TODO: What about phase differences?
        if isinstance(other, SineFunction):
            return self.length
        else:
            return False


class DirichletFunction(base.BaseOscillation):

    def __init__(self, periodicity: int, length: int, amplitude: float = 1.0, periods: int = None):

        self.periodicity = periodicity
        self.length = length
        self.amplitude = amplitude

        # create the periods (at least 5 values per period so we have no aliasing)
        if periods is None:
            self.periods = 4
        else:
            self.periods = periods

        # make a check that we have enough samples for the amount of periods
        if self.length >= periods*10:
            raise ValueError(f"The length must be at least 10 times the periods (For {periods} periods we require "
                             f"at least {periods*10} samples).")


    @property
    def shape(self) -> tuple[int,]:
        return (self.length,)

    def render(self):
        return self.amplitude*spspec.diric(np.linspace(-(self.periods+0.5)*np.pi,
                                                       (self.periods+0.5)*np.pi,
                                                       num=self.length),
                                           self.periodicity)

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about sinus that might be similar?
        # TODO: What about phase differences?
        if isinstance(other, DirichletFunction):
            return ((self.periodicity == other.periodicity)
                    and (self.length == other.length)
                    and (self.amplitude == other.amplitude))
        else:
            return False