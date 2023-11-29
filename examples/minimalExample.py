import numpy as np
import matplotlib.pyplot as plt
from changepoynt.algorithms.esst import ESST
from changepoynt.visualization.score_plotting import plot_data_and_score

# create a signal that goes from steady to exponential decline into a sine curve
exp_signal = np.exp(-np.linspace(0, 5, 200))
steady_after = np.exp(-5)*np.ones(150)
steady_before = np.ones(200)
sine_after = 0.2*np.sin(np.linspace(0, 3*np.pi*10, 300))

# make the signal
signal = np.concatenate((steady_before, exp_signal, steady_after, sine_after))
signal += 0.01*np.random.randn(signal.shape[0])

# make change point detection
detector = ESST(30)
detection = detector.transform(signal)

# make the plot
plot_data_and_score(signal, detection)
plt.show()
