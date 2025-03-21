"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    https://gregorygundersen.com/blog/2019/08/13/bocd/
    https://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
============================================================================"""
import numpy as np
from scipy.stats import norm as scipy_norm
from scipy.special import logsumexp as scipy_logsumexp
from changepoynt.algorithms.base_algorithm import Algorithm


class BOCPD(Algorithm):

    def __init__(self, run_length, prior_mean: float = None, prior_var: float = None, signal_var: float = None,
                 change_length_threshold: int = None):

        # save the parameters
        self.run_length = run_length
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.signal_var = signal_var
        assert isinstance(change_length_threshold, int) or change_length_threshold is None, \
            'Change_length_threshold must be an integer.'
        self.change_length_threshold = change_length_threshold

        # mark that we have not yet called fit
        self.has_fit = False

    def fit(self, time_series: np.ndarray):

        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # compute the starting point of the scoring (past and future hankel need to fit)
        assert time_series.shape[0] >= self.run_length, "The time series is too short to score any points."

        # check whether we need to estimate the prior mean and variance
        sliding_window = np.lib.stride_tricks.sliding_window_view(time_series, self.run_length)
        if self.prior_mean is None:
            self.prior_mean = np.median(np.mean(sliding_window, axis=1))
        if self.prior_var is None:
            self.prior_var = np.var(np.mean(sliding_window, axis=1))

        # check whether we need to estimate the signal var
        if self.signal_var is None:
            self.signal_var = np.median(np.var(sliding_window, axis=1))

        # mark that we have done the fit
        self.has_fit = True
        print(self.signal_var, self.prior_mean, self.prior_var)

    def transform(self, time_series: np.ndarray):
        # TODO: make numba version
        # https://github.com/aesara-devs/aesara/issues/404#issuecomment-1835242816

        # fit that has not been done yet
        if not self.has_fit:
            self.fit(time_series)

        # make the model with unknown mean and a gaussian prior of the mean
        model = GaussianUnknownMean(self.prior_mean, self.prior_var, self.signal_var)

        # compute some parameters
        log_hazard = np.log(1/self.run_length)
        log_one_minus_hazard = np.log(1 - 1/self.run_length)
        log_message = np.array([0])  # log 0 == 1
        change_length_threshold = self.change_length_threshold if self.change_length_threshold is not None else int(self.run_length*0.1)

        # create arrays to save intermediate results
        pmean = np.empty_like(time_series)
        pvar = np.empty_like(time_series)
        log_run_length_prob = -np.inf * np.ones((time_series.shape[0] + 1, time_series.shape[0] + 1))
        log_run_length_prob[:self.run_length-1, 0] = 0  # log 0 == 1
        run_length_prob = np.zeros((time_series.shape[0] + 1, time_series.shape[0] + 1))
        run_length_prob[:self.run_length-1, 0] = 1  # step 0 == 1
        for t in range(1, time_series.shape[0]):

            # 2. Observe new datum
            x = time_series[t-1]

            # Make model predictions
            pmean[t - 1] = np.sum(np.exp(log_run_length_prob[t - 1, :t]) * model.mean_params[:t])
            pvar[t - 1] = np.sum(np.exp(log_run_length_prob[t - 1, :t]) * model.var_params[:t])

            # 3. Evaluate predictive probabilities
            log_pis = model.log_pred_prob(t, x)

            # 4. Calculate growth probabilities
            log_growth_probs = log_pis + log_message + log_one_minus_hazard

            # 5. Calculate changepoint probabilities
            log_cp_prob = scipy_logsumexp(log_pis + log_message + log_hazard)

            # 6. Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)

            # 7. Determine run length distribution
            log_run_length_prob[t, :t+1] = new_log_joint
            log_run_length_prob[t, :t+1] -= scipy_logsumexp(new_log_joint)

            # 8. Update sufficient statistics
            model.update_params(t, x)

            # Pass message.
            log_message = new_log_joint

        # compute the change score probability (which we set as the probability that the run length is smaller
        # than one percent of the defined run_length
        change_score = np.sum(np.exp(log_run_length_prob[1:, :change_length_threshold + 1]), axis=1)
        return change_score




# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T = len(data)
    log_R = -np.inf * np.ones((T + 1, T + 1))
    log_R[0, 0] = 0  # log 0 == 1
    pmean = np.empty(T)  # Model's predictive mean.
    pvar = np.empty(T)  # Model's predictive variance.
    log_message = np.array([0])  # log 0 == 1
    log_H = np.log(hazard)
    log_1mH = np.log(1 - hazard)

    for t in range(1, T + 1):
        # 2. Observe new datum.
        x = data[t - 1]

        # Make model predictions.
        pmean[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.mean_params[:t])
        pvar[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.var_params[:t])

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = scipy_logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t + 1] = new_log_joint
        log_R[t, :t + 1] -= scipy_logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar


# -----------------------------------------------------------------------------


class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.

        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return scipy_norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params = (self.mean_params * self.prec_params[:-1] + (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx


# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data = []
    cps = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, cps, R, pmean, pvar):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))

    ax1, ax2, ax3 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)

    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r',
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        ax2.axvline(cp, c='red', ls='dotted')
        ax3.axvline(cp, c='red', ls='dotted')

    # compute the weighted mean run length
    ax3.plot(range(-1, T), np.sum(R[:, :int(T*0.01)], axis=1))
    ax3.set_xlim([0, T])

    plt.tight_layout()


# -----------------------------------------------------------------------------

def main_test():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from changepoynt.visualization.score_plotting import plot_data_and_score
    T = 1000  # Number of observations.
    hazard = 1 / 100  # Constant prior on changepoint probability.
    mean0 = 0  # The prior mean on the mean parameter.
    var0 = 2  # The prior variance for mean parameter.
    varx = 1  # The known variance of the data.

    data, cps = generate_data(varx, mean0, var0, T, hazard)
    model = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(data, model, hazard)

    plot_posterior(T, data, cps, R, pmean, pvar)

    # make the same using our implementation
    data = np.array(data)
    transf = BOCPD(100, mean0, var0)
    transf = BOCPD(100)
    score = transf.transform(np.array(data))
    plot_data_and_score(np.array(data), score)


    plt.show()

if __name__ == '__main__':
    main_test()