# TODO add tests for the simulation
import pytest
import numpy as np

import changepoynt.simulation as cpsim
import changepoynt.simulation.base as simbase
import changepoynt.simulation.randomizers as rds
import changepoynt.simulation.generator as simgen
from changepoynt.simulation.signals import ChangeSignal


class TestSimulation:
    def setup_method(self):
        # make a random generator
        randg = np.random.default_rng(42)

        # make event generator
        ceg = simgen.ChangeGenerator(length=1000, minimum_length=30, rate=0.01, random_generator=randg,verbose=False)

        # make signal generator
        csg = simgen.ChangeSignalGenerator()

        # create the completely independent signals
        events = []
        for _ in range(10):
            event, _ = ceg.generate_independent_list_disturbed(1, 0)
            events.append(event[0])

        # create the signals
        self.signals = [csg.generate_from_events(event_list) for event_list in events]

    def teardown_method(self):
        pass

    def test_class_naming(self):

        with pytest.raises(ValueError):
            class SineOscillation(simbase.BaseTrend):
                amplitude = simbase.Parameter(float, 1.0,default_parameter_distribution=rds.ContinuousGaussianDistribution(1.0))

                def render(self) -> np.ndarray:
                    return np.sin(np.linspace(start=0.0, stop=np.pi, num=self.length)) * self.amplitude
            a = SineOscillation(100, period=5.0, amplitude=5.0)

    def test_forbidden_class_names(self):

        with pytest.raises(NameError):
            class Signal(simbase.BaseTrend):
                amplitude = simbase.Parameter(float, 1.0,default_parameter_distribution=rds.ContinuousGaussianDistribution(1.0))

                def render(self) -> np.ndarray:
                    return np.sin(np.linspace(start=0.0, stop=np.pi, num=self.length)) * self.amplitude
            a = Signal(100, period=5.0, amplitude=5.0)

    def test_class_reserved_parameters(self):
        with pytest.raises(AttributeError):
            class SineASSDSAFFSFEEWFTESTCLASS(simbase.BaseTrend):
                amplitude = simbase.Parameter(float, 1.0,
                                              default_parameter_distribution=rds.ContinuousGaussianDistribution(1.0))
                type_name = 0

                def render(self) -> np.ndarray:
                    return np.sin(np.linspace(start=0.0, stop=np.pi, num=self.length)) * self.amplitude

    def test_json_type_warnint(self):
        with pytest.warns(UserWarning):
            class SineOscillationTESTCASEBLABLA(simbase.BaseTrend):
                amplitude = simbase.Parameter(np.int32, np.int32(1.0), use_random=False)

                def render(self) -> np.ndarray:
                    return np.sin(np.linspace(start=0.0, stop=np.pi, num=self.length)) * self.amplitude

            a = SineOscillationTESTCASEBLABLA(100, amplitude=np.int32(5))

    def test_json_serialization(self):

        # get a signal from the setup
        signal = self.signals[0]

        # serialize the signal
        signal_str = signal.to_json()

        # deserialize again
        new_signal = ChangeSignal.from_json(signal_str)

        # compare them for equality
        np.testing.assert_array_equal(signal.render(), new_signal.render())

    def test_all_oscillations_implementations(self):
        pass


if __name__ == "__main__":
    pytest.main()
