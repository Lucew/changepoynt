import numpy as np

import changepoynt.simulation.base as base
import changepoynt.simulation.trends as trends
import changepoynt.simulation.oscillations as oscillations


class NoTransition(base.BaseTransition):
    allowed_from = (base.SignalPart,)
    allowed_to = (base.SignalPart,)

    def __init__(self, from_object, to_object, **kwargs):
        # have kwargs so it is not a problem if somebody specifies transition_length=1
        super(NoTransition, self).__init__(1, from_object, to_object)

    def get_transition_values(self) -> np.ndarray:
        return np.concatenate((self.start_y, self.end_y))


class ConstantTrendTransition(base.BaseTransition):
    allowed_from = (trends.ConstantOffset,)
    allowed_to = (trends.ConstantOffset,)

    def get_transition_values(self) -> np.ndarray:

        # define the x-axis with the correct length
        z = np.linspace(-10, 10, self.transition_length * 2)

        # render both objects to get start and
        return (1 / (1 + np.exp(-z)))*(self.end_y[-1]-self.start_y[0]) + self.start_y[0]


class LinearTrendTransition(base.BaseTransition):
    allowed_from = (trends.LinearTrend,)
    allowed_to = (trends.LinearTrend,)

    def get_transition_values(self) -> np.ndarray:
        # https://math.stackexchange.com/a/3506549

        # make sigmoid function
        z = np.linspace(-20, 20, self.transition_length * 2)
        sigmoid = (1 / (1 + np.exp(-z)))

        # get the slopes of both linear trends
        prev_slope = getattr(self.from_object, 'slope')
        future_slope = getattr(self.to_object, 'slope')

        # compute how both linear functions would continue
        prev_continued = np.linspace(1, self.transition_length, self.transition_length)*prev_slope+self.start_y[-1]
        future_continued = -np.linspace(1, self.transition_length, self.transition_length)[::-1]*future_slope+self.end_y[0]

        # combine both the continued and the actual function
        prev_combined = np.concatenate((self.start_y, prev_continued))
        future_combined = np.concatenate((future_continued, self.end_y))

        # import matplotlib.pyplot as plt
        # plt.plot(prev_combined)
        # plt.plot(future_combined)

        return prev_combined*(1-sigmoid) + future_combined*sigmoid


class OscillationRampTransition(base.BaseTransition):
    allowed_from = (oscillations.NoOscillation,)
    allowed_to = (oscillations.Periodic,)

    def get_transition_values(self) -> np.ndarray:

        # make sigmoid function
        z = np.linspace(-2, 5, self.transition_length)
        sigmoid = (1 / (1 + np.exp(-z)))

        # multiply the second part with the sigmoid (which has an amplitude)
        future_part = self.end_y * sigmoid
        return np.concatenate((self.start_y, future_part))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # make transitions between no oscillation and oscillation
    osci1 = oscillations.NoOscillation(1000)
    osci2 = oscillations.SineOscillation(1000, periods=15)
    transition1 = OscillationRampTransition(200, osci1, osci2)
    array = np.concatenate(transition1.apply(osci1.render(), osci2.render()))
    plt.plot(array)
    plt.show()

    # make transition between constant trends
    plt.figure()
    trend1 = trends.ConstantOffset(length=200, offset=12)
    trend2 = trends.ConstantOffset(length=200, offset=5)
    transition1 = ConstantTrendTransition(transition_length=40, from_object=trend1, to_object=trend2)
    transition2 = NoTransition(transition_length=101, from_object=trend1, to_object=trend2)
    concat_signal = np.concatenate((trend1.render(), trend2.render()))
    trans_signal = np.concatenate(transition2.apply(trend1.render(), trend2.render()))
    print(np.all(np.equal(trans_signal, concat_signal))) # Should be true


    array = np.concatenate(transition1.apply(trend1.render(), trend2.render()))
    array2 = np.concatenate(transition2.apply(trend1.render(), trend2.render()))
    plt.plot(array)
    plt.plot(array2)
    plt.show()

    # make transition between linear trends
    plt.figure()
    trend3 = trends.LinearTrend(200, slope=5, offset=0)
    trend4 = trends.LinearTrend(200, slope=35, offset=2000)
    transition3 = LinearTrendTransition(transition_length=100, from_object=trend3, to_object=trend4)
    transition4 = NoTransition(transition_length=20, from_object=trend3, to_object=trend4)
    concat_signal = np.concatenate((trend3.render(), trend4.render()))
    trans_signal = np.concatenate(transition4.apply(trend3.render(), trend4.render()))
    print(np.all(np.equal(trans_signal, concat_signal)))  # Should be true

    array = np.concatenate(transition3.apply(trend3.render(), trend4.render()))
    array2 = np.concatenate(transition4.apply(trend3.render(), trend4.render()))
    plt.plot(array)
    plt.plot(array2)
    plt.show()

    print(base.SignalPart.get_possible_transitions(trend1.__class__, trend2.__class__))
    print(base.SignalPart.get_all_possible_transitions())
