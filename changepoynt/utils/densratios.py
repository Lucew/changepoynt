"""
We use an implemented package called densratio for the computations of the density ratio and the Pearson divergence
calculations. We merely adopt some steps to account for the symmetric Pearson divergence used in [1]. We adopted
the function RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num=100, verbose=True) from the pacakge.

[1]
Liu, Song, et al.
"Change-point detection in time-series data by relative density-ratio estimation."
Neural Networks 43 (2013): 72-83.

The code has simply been adopted from: densratio 0.3.0:
- https://github.com/hoxo-m/densratio_py
- https://pypi.org/project/densratio/
published using an MIT license.

We can not thank the authors enough for writing such deep mathematical code!
The reason for adoption were the one named above (symmetric pearson divergence)
and more direct use of numba as a jit compiler, as well as we do not want and use many things computed in the
package, so we can save time by not computing them.
"""
import numpy as np
import numba as nb
from numpy.random import randint
from numpy.linalg import solve
from typing import Union

# define some datatypes
_np_float = np.result_type(float)
_nb_float = nb.from_dtype(_np_float)


def rulsif(x, y, alpha=0.0, sigma_range: Union[str, list[float]] = "auto",
           lambda_range: Union[str, list[float]] = "auto", kernel_num=100):
    """
    Estimation of the alpha-Relative Density Ratio p(x)/p_alpha(x) by RuLSIF
    (Relative Unconstrained Least-Square Importance Fitting) from

    Liu, Song, et al.
    "Change-point detection in time-series data by relative density-ratio estimation."
    Neural Networks 43 (2013): 72-83.

    p_alpha(x) = alpha * p(x) + (1 - alpha) * q(x)

    :param x: Sample from p(x)
    :param y: Sample from q(x)
    :param alpha: Mixture parameter
    :param sigma_range: Search range of Gaussian kernel bandwidth
    :param lambda_range: Search range of regularization parameter
    :param kernel_num: Number of kernels
    :return:
    """

    # transform the input vectors
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be same dimensions.")

    if isinstance(sigma_range, str) and sigma_range != "auto":
        raise TypeError("Invalid value for sigma_range.")

    if isinstance(lambda_range, str) and lambda_range != "auto":
        raise TypeError("Invalid value for lambda_range.")

    if sigma_range is None or (isinstance(sigma_range, str) and sigma_range == "auto"):
        sigma_range = 10 ** np.linspace(-3, 9, 13)

    if lambda_range is None or (isinstance(lambda_range, str) and lambda_range == "auto"):
        lambda_range = 10 ** np.linspace(-3, 9, 13)

    result = _rulsif(x, y, alpha, sigma_range, lambda_range, kernel_num)
    return result


def _rulsif(x: np.ndarray, y: np.ndarray, alpha: float, sigma_range: list[float], lambda_range: list[float],
            kernel_num: int = 100):
    """
    Estimation of the alpha-Relative Density Ratio p(x)/p_alpha(x) by RuLSIF
    (Relative Unconstrained Least-Square Importance Fitting) from

    Liu, Song, et al.
    "Change-point detection in time-series data by relative density-ratio estimation."
    Neural Networks 43 (2013): 72-83.

    p_alpha(x) = alpha * p(x) + (1 - alpha) * q(x)

    :param x: Sample from p(x)
    :param y: Sample from q(x)
    :param alpha: Mixture parameter
    :param sigma_range: Search range of Gaussian kernel bandwidth
    :param lambda_range: Search range of regularization parameter
    :param kernel_num: Number of kernels
    :return:
    """

    # Number of samples.
    nx = x.shape[0]
    ny = y.shape[0]

    # Number of kernel functions.
    kernel_num = min(kernel_num, nx)

    # Randomly take a subset of x, to identify centers for the kernels.
    centers = x[randint(nx, size=kernel_num)]

    if len(sigma_range) == 1 and len(lambda_range) == 1:
        # just use the given sigma and lambda
        sigma = sigma_range[0]
        lambda_ = lambda_range[0]
    else:
        # Grid-search cross-validation for optimal kernel and regularization parameters.
        sigma, lambda_ = search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range)

    phi_x = compute_kernel_gaussian(x, centers, sigma)
    phi_y = compute_kernel_gaussian(y, centers, sigma)
    H = alpha * (phi_x.T.dot(phi_x) / nx) + (1 - alpha) * (phi_y.T.dot(phi_y) / ny)
    h = phi_x.mean(axis=0).T
    theta = np.asarray(solve(H + np.diag(np.array(lambda_).repeat(kernel_num)), h)).ravel()

    # No negative coefficients.
    theta[theta < 0] = 0

    alpha_pe = alpha_pe_divergence(x, y, centers, sigma, theta, alpha)
    alpha_kl = alpha_kl_divergence(x, y, centers, sigma, theta)

    return alpha_kl, alpha_pe, lambda_, sigma


# Compute the alpha-relative density ratio, at the given coordinates.
def compute_alpha_density_ratio(coordinates, centers, sigma, theta):
    # Evaluate the kernel at these coordinates, and take the dot-product with the weights.
    phi_x = compute_kernel_gaussian(coordinates, centers, sigma)
    alpha_density_ratio = phi_x @ theta

    return alpha_density_ratio


# Compute the approximate alpha-relative PE-divergence, given samples x and y from the respective distributions.
def alpha_pe_divergence(x, y, centers, sigma, theta, alpha):
    # x is Y, in Reference [1], y is Y' in reference [1]
    # and this computes symmetric pearson divergence

    # Obtain alpha-relative density ratio at these points.
    g_x = compute_alpha_density_ratio(x, centers, sigma, theta)

    # Obtain alpha-relative density ratio at these points.
    g_y = compute_alpha_density_ratio(y, centers, sigma, theta)

    # Compute the alpha-relative PE-divergence as given in Reference 1.
    n = x.shape[0]
    divergence = (-alpha * (g_x @ g_x)/2 - (1 - alpha) * (g_y @ g_y)/2 + g_x.sum(axis=0))/n - 1./2
    return divergence


# Compute the approximate alpha-relative KL-divergence, given samples x and y from the respective distributions.
def alpha_kl_divergence(x: np.ndarray, y: np.ndarray, centers, sigma, theta):
    # x is Y, in Reference [1]

    # Obtain alpha-relative density ratio at these points.
    g_x = compute_alpha_density_ratio(x, centers, sigma, theta)

    # Compute the alpha-relative KL-divergence.
    n = x.shape[0]
    divergence = np.log(g_x).sum(axis=0) / n
    return divergence


@nb.jit(nopython=True)
def search_sigma_and_lambda(x, y, alpha, centers, sigma_range, lambda_range):
    """
    Grid-search cross-validation for the optimal parameters sigma and lambda by leave-one-out cross-validation as
    proposed in

    "A Least-squares Approach to Direct Importance Estimation"
    Takafumi Kanamori, Shohei Hido, and Masashi Sugiyama,
    Journal of Machine Learning Research 10 (2009) 1391-1445.

    :param x: Sample from p(x)
    :param y: Sample from q(x)
    :param centers: the centers for the different kernels we will be using
    :param alpha: Mixture parameter
    :param sigma_range: Search range of Gaussian kernel bandwidth
    :param lambda_range: Search range of regularization parameter
    :return:
    """
    nx = x.shape[0]
    ny = y.shape[0]
    n_min = min(nx, ny)
    kernel_num = centers.shape[0]

    score_new = np.inf
    sigma_new = 0
    lambda_new = 0

    for sigma in sigma_range:

        phi_x = compute_kernel_gaussian(x, centers, sigma)  # (nx, kernel_num)
        phi_y = compute_kernel_gaussian(y, centers, sigma)  # (ny, kernel_num)
        H = alpha * (phi_x.T @ phi_x / nx) + (1 - alpha) * (phi_y.T @ phi_y / ny)  # (kernel_num, kernel_num)
        rows = phi_x.shape[0]
        h = (phi_x.sum(axis=0)/rows).reshape(-1, 1)  # (kernel_num, 1)
        phi_x = phi_x[:n_min].T  # (kernel_num, n_min)
        phi_y = phi_y[:n_min].T  # (kernel_num, n_min)

        for lambda_ in lambda_range:
            B = H + np.diag(np.array(lambda_ * (ny - 1) / ny).repeat(kernel_num))  # (kernel_num, kernel_num)
            B_inv_X = solve(B, phi_y)  # (kernel_num, n_min)
            X_B_inv_X = np.multiply(phi_y, B_inv_X)  # (kernel_num, n_min)
            denom = ny * np.ones(n_min) - np.ones(kernel_num) @ X_B_inv_X  # (n_min, )
            B0 = solve(B, h @ np.ones((1, n_min))) + B_inv_X @ np.diag((h.T @ B_inv_X / denom).flatten())  # (kernel_num, n_min)
            B1 = solve(B, phi_x) + B_inv_X @ np.diag((np.ones(kernel_num) @ np.multiply(phi_x, B_inv_X)).flatten())  # (kernel_num, n_min)
            B2 = (ny - 1) * (nx * B0 - B1) / (ny * (nx - 1))  # (kernel_num, n_min)
            B2 = np.clip(B2, a_min=0, a_max=None)
            r_y = np.multiply(phi_y, B2).sum(axis=0).T  # (n_min, )
            r_x = np.multiply(phi_x, B2).sum(axis=0).T  # (n_min, )

            # Squared loss of RuLSIF, without regularization term.
            # Directly related to the negative of the PE-divergence.
            score = (r_y @ r_y / 2 - r_x.sum(axis=0)) / n_min

            if score < score_new:
                score_new = score
                sigma_new = sigma
                lambda_new = lambda_

    return sigma_new, lambda_new


@nb.guvectorize([nb.void(_nb_float[:, :], _nb_float[:], _nb_float, _nb_float[:])], '(m, p),(p),()->(m)',
                nopython=True, target='cpu', cache=True)
def _compute_kernel_gaussian(x_list, y_row, neg_gamma, res) -> None:
    sq_norm = np.sum(np.power(x_list - y_row, 2), 1)
    np.multiply(neg_gamma, sq_norm, res)
    np.exp(res, res)


@nb.jit(nopython=True)
def _compute_kernel_gaussian2(x_list, y_list, neg_gamma):
    res = np.empty((y_list.shape[0], x_list.shape[0]), _np_float)

    for j, y_row in enumerate(y_list):
        sq_norm = np.sum(np.power(x_list - y_row, 2), 1)
        np.multiply(neg_gamma, sq_norm, res[j].T)
        np.exp(res[j].T, res[j].T)

    return res


# Returns a 2D numpy matrix of kernel evaluated at the grid points with coordinates from x_list and y_list.
@nb.jit(nopython=True)
def compute_kernel_gaussian(x_list, y_list, sigma):
    return _compute_kernel_gaussian2(x_list, y_list, -.5 * sigma ** -2).T


def main():
    # testcase from: https://github.com/hoxo-m/densratio_py
    from scipy.stats import norm
    from time import time
    from densratio import densratio

    np.random.seed(1)
    x = norm.rvs(size=[500, 2], loc=[0, 0], scale=[1. / 8, 1. / 7])
    y = norm.rvs(size=[500, 2], loc=[0, 0], scale=[1. / 2, 1. / 3])
    alpha = 0.1
    kl_divergence, symmetric_pe_divergence, sigma_result, lambda_result = rulsif(x, y, alpha=alpha)
    print(kl_divergence, symmetric_pe_divergence)  # expected: ~0.7037648129307483, ~0.618794133598705
    start = time()
    kl_divergence, symmetric_pe_divergence, sigma_result, lambda_result = rulsif(x, y, alpha=alpha)
    print(kl_divergence, symmetric_pe_divergence, time() - start)  # expected: ~0.7037648129307483, ~0.618794133598705
    start = time()
    kl_divergence, symmetric_pe_divergence, sigma_result, lambda_result = rulsif(x, y, alpha=alpha,
                                                                                 sigma_range=[sigma_result],
                                                                                 lambda_range=[lambda_result])
    print(kl_divergence, symmetric_pe_divergence, time() - start)  # expected: ~0.7037648129307483, ~0.618794133598705
    densratio(x, y, alpha=alpha, verbose=False)
    start = time()
    _ = densratio(x, y, alpha=alpha, verbose=False)
    print(time() - start)  # expected: ~0.7037648129307483, ~0.618794133598705


if __name__ == '__main__':
    main()
