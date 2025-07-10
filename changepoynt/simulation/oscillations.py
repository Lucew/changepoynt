from abc import abstractmethod

import numpy as np
import scipy.signal as spsig
import scipy.special as spspec

from changepoynt.simulation import base
import changepoynt.simulation.randomizers as rds


class NoOscillation(base.BaseOscillation):
    """
    This will implement a simple line. Offset will be done using a trend.
    """
    def render(self) -> np.ndarray:
        return np.zeros(self.shape)


class Periodic(base.BaseOscillation):
    periods = base.Parameter(int, limit=(1, 100), tolerance=0.5, use_for_comparison=False, default_parameter_distribution=rds.DiscretePoissonDistribution(20, 1, 100))
    amplitude = base.Parameter((float, int), limit=(-np.inf, np.inf), tolerance=0.1, default_value=1.0, default_parameter_distribution=rds.ContinuousConditionalGaussianDistribution(standard_deviation=2, default_mean=0.5))
    wavelength = base.Parameter(float, limit=(5.0, np.inf), derived=True, tolerance=0.1, modifiable=False, use_random=False, limit_error_explanation="We require at least 5 samples per period. Either specify less periods or greater length.")

    def compute_wavelength(self):
        return self.length / self.periods

    @abstractmethod
    def render(self):
        raise NotImplementedError

class SineOscillation(Periodic):

    def render(self) -> np.ndarray:
        return self.amplitude * np.sin(np.linspace(0, self.periods * np.pi * 2, self.length))


class DirichletOscillation(Periodic):
    periodicity = base.Parameter(int, limit=(1, np.inf), tolerance=1, default_parameter_distribution=rds.DiscreteConditionalGaussianDistribution(1, minimum=1, default_mean=2.0))

    @staticmethod
    def compute_start_end(periods, periodicity):

        # Determine fundamental period based on periodicity
        if periodicity & 1 == 0:
            # Even n => fundamental period is 4*pi
            fundamental_period = 4 * np.pi
        else:
            # Odd n => fundamental period is 2*pi
            fundamental_period = 2 * np.pi

        # Total range to cover the requested number of periods
        total_range = periods * fundamental_period

        # Offset to the first zero crossing: x_zero = 2*pi/periodicity
        zero_offset = 2 * np.pi / periodicity
        return zero_offset, total_range

    def render(self) -> np.ndarray:
        # compute the offset and the end value so we have zeros at both ends
        zero_offset, total_range = self.compute_start_end(self.periods, self.periodicity)

        # Sample points from first zero crossing to last, inclusive
        start = zero_offset
        end = start + total_range
        x = np.linspace(start, end, num=self.length, endpoint=True)

        # render the function
        return spspec.diric(x, self.periodicity)*self.amplitude


class SquareOscillation(Periodic):
    duty = base.Parameter((float, int), limit=(0, 1), tolerance=0.05, default_value=0.5, default_parameter_distribution=rds.ContinuousGaussianDistribution(0.1, 0.5, 0.0, 1.0))
    wavelength = base.Parameter(float, limit=(2, np.inf), derived=True, tolerance=0.1, modifiable=False, use_random=False, limit_error_explanation="We require at least 2 samples per period. Either specify less periods or greater length.")

    def render(self) -> np.ndarray:
        # start at -1 and end at -1 (half of a period multiplied with duty)
        x = np.linspace(0, self.periods * np.pi*2, self.length)

        # elevate to oscillate between zero and one and multiple with amplitude
        output = self.amplitude*spsig.square(x, self.duty)
        output[0] = 0
        output[-1] = 0
        return output


class SawtoothOscillation(Periodic):
    width = base.Parameter((float, int), limit=(0, 1), tolerance=0.05, default_parameter_distribution=rds.ContinuousUniformDistribution(0.0, 1.0))
    wavelength = base.Parameter(float, limit=(6, np.inf), derived=True, tolerance=0.1, use_random=False, modifiable=False, limit_error_explanation="We require at least 6 samples per period. Either specify less periods or greater length.")

    def render(self) -> np.ndarray:
        # start at -1 and end at -1 (half of a period multiplied with duty)
        x = np.linspace(np.pi*self.width, self.periods * np.pi*2 + self.width*np.pi, self.length)

        # elevate to oscillate between zero and one and multiple with amplitude
        return self.amplitude*spsig.sawtooth(x, self.width)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # check the dirichlet function
    plt.figure()
    diric1 = DirichletOscillation(length=500, periods=10, periodicity=5)
    diric2 = DirichletOscillation(length=1000, periods=5, periodicity=5)
    print('Equality for Dirichlet', diric1 == diric2, diric2==diric1)
    plt.plot(diric1.render())
    func = diric1.render()
    print(func[0], func[-1])  # should be zero
    print('ampl')
    print(diric1.amplitude)
    diric1.amplitude *= 2
    print(diric1.amplitude)
    plt.plot(diric1.render())


    # check the sine function
    plt.figure()
    sine1 = SineOscillation(length=500, periods=10)
    sine2 = SineOscillation(length=1000, periods=5)
    print('Equality for Sine', sine1 == sine2, sine2 == sine1, sine1 == sine1)
    plt.plot(sine1.render())
    func = sine1.render()
    print(func[0], func[-1])  # should be zero

    # check the square function
    plt.figure()
    square1 = SquareOscillation(length=1000, periods=10, duty=0.9)
    square2 = SquareOscillation(length=1000, periods=5)
    print('Equality for Square', square1 == square2, square2 == square1)
    plt.plot(square1.render())
    plt.plot(square2.render())
    func = square1.render()
    print(func[0], func[-1])  # should be zero

    # check the sawtooth function
    plt.figure()
    sawtooth1 = SawtoothOscillation(length=1000, periods=10, width=1.0)
    sawtooth2 = SawtoothOscillation(length=1000, periods=5, width=0.78)
    print('Equality for Sawtooth', sawtooth1 == sawtooth2, sawtooth2 == sawtooth1)
    plt.plot(sawtooth1.render())
    plt.plot(sawtooth2.render())
    func = sawtooth1.render()
    print(func[0], func[-1])  # should be zero
    print(sawtooth1 == square1)
    plt.show()
    print(SawtoothOscillation.get_parameters_for_randomizations())
