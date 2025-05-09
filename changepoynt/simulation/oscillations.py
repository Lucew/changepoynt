import numpy as np
import scipy.signal as spsig
import scipy.special as spspec

import base

# TODO: line. parameters: None. We will use the trend for further modifications
# TODO: ecg? but this introduces other dependencies
# TODO: check change point papers for what they simulate


class NoOscillation(base.BaseOscillation):
    """
    This will implement a simple line. Offset will be done using a trend.
    """
    def __init__(self, length: int):
        super().__init__(length, 0)
        self.length = length


    def render(self) -> np.ndarray:
        return np.zeros(self.length)

    def __eq__(self, other):

        # check whether the other series is of the same class
        return isinstance(other, type(self))


class SineOscillation(base.BaseOscillation):

    def __init__(self, length: int , periods: int = None, amplitude: float = 1.0, tolerance: float = 1.0):
        super().__init__(length, tolerance)

        # save the variables
        self.amplitude = amplitude
        self.periods = periods

        # check that each period contains at least 5 samples
        if periods is None:
            self.periods = self.length//5
        if self.length < self.periods*5:
            raise ValueError("We require at least 5 samples per period")

    def render(self):
        return self.amplitude*np.sin(np.linspace(0, self.periods * np.pi*2, self.length))

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about diric that might be similar?
        if isinstance(other, type(self)):

            # compute how many samples per period both have
            frequency_self = self.length/self.periods
            frequency_other = other.length/other.periods

            # check for equality with tolerance
            frequency_equal = abs(frequency_other-frequency_self) < self.tolerance
            amplitude_equal = abs(other.amplitude-self.amplitude) < self.tolerance
            return frequency_equal and amplitude_equal
        else:
            return False


class DirichletOscillation(base.BaseOscillation):

    def __init__(self, length: int, periods: int = None, periodicity: int = 5, amplitude: float = 1.0,
                 tolerance: float = 0.05):
        super().__init__(length, tolerance)

        self.periods = periods
        self.periodicity = periodicity
        self.amplitude = amplitude

        # set the default value for the periods
        if periods is None:
            self.periods = self.length//10

        # check the correct values
        if self.length < self.periods * 10:
            raise ValueError(
                f"length ({self.length}) must be at least 10 samples per period: "
                f"min required {self.periods * 10}.")
        if self.periodicity < 1:
            raise ValueError("periodicity must be a positive integer.")

        # Determine fundamental period based on periodicity
        if self.periodicity&1 == 0:
            # Even n => fundamental period is 4*pi
            self.fundamental_period = 4 * np.pi
        else:
            # Odd n => fundamental period is 2*pi
            self.fundamental_period = 2 * np.pi

        # Total range to cover the requested number of periods
        self.total_range = self.periods * self.fundamental_period

        # Offset to the first zero crossing: x_zero = 2*pi/periodicity
        self.zero_offset = 2 * np.pi / self.periodicity

    def render(self):
        # Sample points from first zero crossing to last, inclusive
        start = self.zero_offset
        end = start + self.total_range
        x = np.linspace(start, end, num=self.length, endpoint=True)
        return spspec.diric(x, self.periodicity)*self.amplitude

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about sinus that might be similar?
        # TODO: What about phase differences?
        if isinstance(other, type(self)):

            # compute the frequencies
            frequency_self = self.length / self.periods
            frequency_other = other.length / other.periods

            # check for equal frequencies, amplitudes, and periodicity
            frequency_equal = abs(frequency_other - frequency_self) < self.tolerance
            amplitude_equal = abs(other.amplitude - self.amplitude) < self.tolerance
            periodicity_equal = abs(other.periodicity - other.periodicity) < self.tolerance
            return frequency_equal and amplitude_equal and periodicity_equal
        else:
            return False


class SquareOscillation(base.BaseOscillation):

    def __init__(self, length: int , periods: int = None, duty: float = 0.5, amplitude: float = 1.0,
                 tolerance: float = 1.0):
        super().__init__(length, tolerance)

        # save the variables
        self.amplitude = amplitude
        self.periods = periods
        self.duty = duty

        # check that each period contains at least 6 samples
        if periods is None:
            self.periods = self.length//6
        if self.length < self.periods*6:
            raise ValueError("We require at least 6 samples per period")

        # check that the duty is between zero and one
        if self.duty < 0 or self.duty > 1:
            raise ValueError(f"duty must be between 0 and 1. Currently it is: {self.duty}.")

    def render(self):
        # start at -1 and end at -1 (half of a period multiplied with duty)
        x = np.linspace(0, self.periods * np.pi*2, self.length)

        # elevate to oscillate between zero and one and multiple with amplitude
        output = self.amplitude*spsig.square(x, self.duty)
        output[0] = 0
        output[-1] = 0
        return output

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about diric that might be similar?
        if isinstance(other, type(self)):

            # compute how many samples per period both have
            frequency_self = self.length/self.periods
            frequency_other = other.length/other.periods

            # check for equality with tolerance
            frequency_equal = abs(frequency_other-frequency_self) < self.tolerance
            amplitude_equal = abs(other.amplitude-self.amplitude) < self.tolerance
            return frequency_equal and amplitude_equal
        else:
            return False


class SawtoothOscillation(base.BaseOscillation):

    def __init__(self, length: int , periods: int = None, width: float = 0.5, amplitude: float = 1.0,
                 tolerance: float = 1.0):
        super().__init__(length, tolerance)

        # save the variables
        self.amplitude = amplitude
        self.periods = periods
        self.width = width

        # check that each period contains at least 6 samples
        if periods is None:
            self.periods = self.length//6
        if self.length < self.periods*6:
            raise ValueError("We require at least 6 samples per period")

        # check that the duty is between zero and one
        if self.width < 0 or self.width > 1:
            raise ValueError(f"width must be between 0 and 1. Currently it is: {self.width}.")

    def render(self):
        # start at -1 and end at -1 (half of a period multiplied with duty)
        x = np.linspace(np.pi*self.width, self.periods * np.pi*2 + self.width*np.pi, self.length)

        # elevate to oscillate between zero and one and multiple with amplitude
        return self.amplitude*spsig.sawtooth(x, self.width)

    def __eq__(self, other):

        # check whether the other series is of the same class
        # TODO: What about diric that might be similar?
        if isinstance(other, type(self)):

            # compute how many samples per period both have
            frequency_self = self.length/self.periods
            frequency_other = other.length/other.periods

            # check for equality with tolerance
            frequency_equal = abs(frequency_other-frequency_self) < self.tolerance
            amplitude_equal = abs(other.amplitude-self.amplitude) < self.tolerance
            return frequency_equal and amplitude_equal
        else:
            return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # check the dirichlet function
    plt.figure()
    diric1 = DirichletOscillation(length=500, periods=10, periodicity=5)
    diric2 = DirichletOscillation(length=1000, periods=5, periodicity=5)
    print(diric1 == diric2, diric2==diric1)
    plt.plot(diric1.render())
    func = diric1.render()
    print(func[0], func[-1])  # should be zero

    # check the sine function
    plt.figure()
    sine1 = SineOscillation(length=500, periods=10)
    sine2 = SineOscillation(length=1000, periods=5)
    print(sine1 == sine2, sine2 == sine1)
    plt.plot(sine1.render())
    func = sine1.render()
    print(func[0], func[-1])  # should be zero

    # check the square function
    plt.figure()
    square1 = SquareOscillation(length=1000, periods=10)
    square2 = SquareOscillation(length=1000, periods=5)
    print(square1 == square2, square2 == square1)
    plt.plot(square1.render())
    plt.plot(square2.render())
    func = square1.render()
    print(func[0], func[-1])  # should be zero

    # check the sawtooth function
    plt.figure()
    sawtooth1 = SawtoothOscillation(length=1000, periods=10, width=1)
    sawtooth2 = SawtoothOscillation(length=1000, periods=5)
    print(sawtooth1 == sawtooth2, sawtooth2 == sawtooth1)
    plt.plot(sawtooth1.render())
    plt.plot(sawtooth2.render())
    func = sawtooth1.render()
    print(func[0], func[-1])  # should be zero

    print(sawtooth1 == square1)

    plt.show()