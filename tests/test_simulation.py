# TODO add tests for the simulation
import pytest
import numpy as np

import changepoynt.simulation.base as simbase
import changepoynt.simulation.randomizers as rds
import changepoynt.simulation.generator as simgen
import changepoynt.simulation.signals as simsig
import changepoynt.simulation.serialization as simser


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
        for _ in range(50):
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
            del simbase.SignalPart._registry[a.__class__.__name__]
            del a
            del SineOscillationTESTCASEBLABLA

    def test_json_serialization(self):

        # get a signal from the setup
        signal = self.signals[0]

        # serialize the signal
        signal_str = signal.to_json()

        # deserialize again
        new_signal = simsig.ChangeSignal.from_json(signal_str)

        # compare them for equality
        np.testing.assert_array_equal(signal.render(), new_signal.render())

        # create a multivariate change point signal
        multisig = simsig.ChangeSignalMultivariate(self.signals, [str(idx) for idx,_ in enumerate(self.signals)])
        json_str = multisig.to_json()
        new_multisig = simsig.ChangeSignalMultivariate.from_json(json_str)
        np.testing.assert_array_equal(multisig.render(), new_multisig.render())

    def test_json_generic_serialization(self):

        # get a signal from the setup
        signal = self.signals[0]

        # serialize and deserialize the signal
        signal_str = simser.to_json(signal)
        assert type(signal_str) == str
        print(type(signal_str))
        new_signal = simser.from_json(signal_str)

        # compare them for equality
        np.testing.assert_array_equal(signal.render(), new_signal.render())

        # create a multivariate change point signal
        multisig = simsig.ChangeSignalMultivariate(self.signals, [str(idx) for idx,_ in enumerate(self.signals)])

        # serialize and deserialize
        json_str = simser.to_json(multisig)
        new_multisig = simser.from_json(json_str)

        # check whether it is still the same signal
        np.testing.assert_array_equal(multisig.render(), new_multisig.render())

    def test_copy(self):

        # get a signal from the setup
        signal = self.signals[0]

        # make a copy of the signal
        signal_copy = signal.copy()
        print(signal)

        # check for equality
        assert signal == signal_copy

        # compare their renders for equality
        np.testing.assert_array_equal(signal.render(), signal_copy.render())

        # check whether they are the same object
        assert not signal is signal_copy

    def test_concatenate(self):

        # get a signal from the setup
        signal = self.signals[0]
        signal2 = self.signals[1]

        # concatenate the signals
        signal_concat = signal.concatenate(signal2)

        # serialize the concatenated signal
        signal_concat_copy = signal_concat.copy()

        # check for equality of the concatenated signals with the original signals
        np.testing.assert_array_equal(signal_concat.render(), np.concatenate((signal.render(), signal2.render())))

        # check whether serialization still works
        signal_concat.render()
        signal_concat_copy.render()
        np.testing.assert_array_equal(signal_concat.render(), signal_concat_copy.render())

    def test_extend(self):

        # get a signal from the setup
        signal_concat = self.signals[0]

        # extend the complete other signals
        signal_concat = signal_concat.extend(self.signals[1:])

        # create the other array by using numpy concatenate
        other_array = np.concatenate(tuple(sig.render() for sig in self.signals))

        # check for equality of the concatenated signals with the original signals
        np.testing.assert_array_equal(signal_concat.render(), other_array)

        # check whether serialization still works
        signal_concat.render()
        signal_concat_copy = signal_concat.copy()
        signal_concat_copy.render()
        np.testing.assert_array_equal(signal_concat.render(), signal_concat_copy.render())

    def test_all_oscillations_implementations(self):
        pass


if __name__ == "__main__":
    pytest.main()
