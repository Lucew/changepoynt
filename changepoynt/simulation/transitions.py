import numpy as np
import scipy.interpolate as spinter

import changepoynt.simulation.base as base
import changepoynt.simulation.trends as trends


class NoTransition(base.BaseTransition):
    allowed_from = (base.SignalPart,)
    allowed_to = (base.SignalPart,)

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # make transition between constant trends
    trend1 = trends.ConstantOffset(length=200, offset=12)
    trend2 = trends.ConstantOffset(length=200, offset=5)
    transition1 = ConstantTrendTransition(transition_length=40, from_object=trend1, to_object=trend2)
    transition2 = NoTransition(transition_length=100, from_object=trend1, to_object=trend2)
    print(np.all(np.equal(transition2.render(), np.concatenate((transition2.from_object.render(), transition2.to_object.render()))))) # has to be true

    array = transition1.render()
    array2 = transition2.render()
    plt.plot(array)
    plt.plot(array2)
    plt.show()

    # make transition between linear trends
    plt.figure()
    trend3 = trends.LinearTrend(200, slope=5, offset=0)
    trend4 = trends.LinearTrend(200, slope=35, offset=10)
    transition3 = LinearTrendTransition(transition_length=100, from_object=trend3, to_object=trend4)
    transition4 = NoTransition(transition_length=100, from_object=trend3, to_object=trend4)
    print(np.all(np.equal(transition4.render(), np.concatenate((transition4.from_object.render(), transition4.to_object.render()))))) # has to be true
    array = transition3.render()
    array2 = transition4.render()
    plt.plot(array)
    # plt.plot(array2)
    plt.show()

    print(base.SignalPart.get_possible_transitions(trend1, trend2))
