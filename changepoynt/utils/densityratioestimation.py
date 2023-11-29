"""
Relative Unconstrained Least-Squares Fitting (RuLSIF): A Python Implementation

[1] Liu S , Yamada M , Collier N , et al.
Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation[J].
2012.

[2] Kawahara Y , Sugiyama M .
Sequential change‐point detection based on direct density‐ratio estimation[M].
John Wiley & Sons, Inc. 2012.

[3] Kawahara Y , Yairi T , Machida K .
Change-Point Detection in Time-Series Data Based on Subspace Identification[C]
Data Mining, 2007. ICDM 2007. Seventh IEEE International Conference on. IEEE, 2007.

Taken from https://github.com/chenxingqiang/rulsif_abrupt-change_detection
copied and heavily modified for performance and jit compilation.
Also corrected a missing transpose in the Gaussian Kernel class.
Additionally, optimized the cross validation to not recompute the gaussian kernel several times
as we only change a factor within the exponent (the sigma).

Has been compared to the demo implementation of the original author:
https://github.com/anewgithubname/change_detection

Unfortunately, the cross validation can't be JIT compiled, as we need to compute theta_hat, which is way faster
using a specialized scipy function for hermitian matrices. Numba can't deal with scipy yet.
"""
import numpy as np
import scipy
import numba as nb
from warnings import warn
from scipy.linalg._misc import LinAlgError, LinAlgWarning
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork


def _solve_check(info, lamch=None, rcond=None):
    """
    Stolen from scipy to use scipy.linalg.solve
    Check arguments during the different steps of the solution phase
    """
    if info < 0:
        raise ValueError('LAPACK reported an illegal value in {}-th argument'
                         '.'.format(-info))
    elif 0 < info:
        raise LinAlgError('Matrix is singular.')

    if lamch is None:
        return
    E = lamch('E')
    if rcond < E:
        warn('Ill-conditioned matrix (rcond={:.6g}): '
             'result may not be accurate.'.format(rcond),
             LinAlgWarning, stacklevel=3)


def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False):
    """
    DANGER!
    This implementation is stolen from within the scipy package and is equivalent to scipy.linalg.solve
    We reduced a lot of checks to tailor it to the given case within this project and make it as fast as possible
    as it is in the hottest path of RuLSIF and by far the *most called function*.

    This might result in problems, as the checks are really minimal.
    """

    # calculate matrix norm using LAPACK
    lange = get_lapack_funcs('lange', (a,))
    anorm = lange('1', a)

    # this is the code stolen from scipy.linalg.solve for the option assume_a = "sym"
    sycon, sysv, sysv_lw = get_lapack_funcs(('sycon', 'sysv', 'sysv_lwork'), (a, b))
    lu, ipvt, x, info = sysv(a, b, lwork=_compute_lwork(sysv_lw, a.shape[0], lower),
                             lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b)

    # check whether the solver worked in LAPACK
    _solve_check(info)

    # can't tell what this is, but seems like we need it
    rcond, info = sycon(lu, ipvt, anorm)

    # Get the correct lamch function for doubles (can't use any other!)
    lamch = get_lapack_funcs('lamch', dtype='d')
    _solve_check(info, lamch, rcond)
    return x


@nb.njit()
def compute_distance(samples: np.ndarray, sample_means: np.ndarray) -> np.ndarray:
    """
    Compute the distances between points in the sample's feature space
    to points along the center of the distribution
    """

    # get the amount of samples
    (sd, sample_cols) = samples.shape
    (md, mean_cols) = sample_means.shape

    # compute the squared sums
    squared_samples = np.sum(samples ** 2, 0)
    squared_means = np.sum(sample_means ** 2, 0)

    # compute the kernel matrix by using (a-b)^2 = a^2 - 2ab + b^2
    """kernel_matrix = np.tile(squared_means, (sample_cols, 1))\
                    + np.tile(squared_samples, (mean_cols, 1)).T\
                    - 2 * np.dot(samples.T, sample_means)"""
    repeated_squared_samples = np.zeros((mean_cols, sample_cols))
    for idx in range(mean_cols):
        repeated_squared_samples[idx, :] = squared_samples
    repeated_squared_means = np.zeros((sample_cols, mean_cols))
    for idx in range(sample_cols):
        repeated_squared_means[idx, :] = squared_means
    kernel_matrix = repeated_squared_means + repeated_squared_samples.T - 2 * samples.T@sample_means

    return kernel_matrix


@nb.njit()
def compute_gaussian_kernel(samples: np.ndarray, sample_means: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes an n-dimensional Gaussian/RBF kernel matrix by taking points
    in the sample's feature space and maps them to kernel coordinates in
    Hilbert space by calculating the distance to each point in the sample
    space and taking the Gaussian function of the distances.
       K(X,Y) = exp( -(|| X - Y ||^2) / (2 * sigma^2) )
    where X is the matrix of data points in the sample space,
          Y is the matrix of gaussian centers in the sample space
         sigma is the width of the gaussian function being used
    """
    squared_distance = compute_distance(samples, sample_means)
    # squared_distance = scipy.spatial.distance_matrix(samples.T, sample_means.T)**2
    return np.exp(-squared_distance / (2 * (sigma ** 2)))


@nb.njit()
def update_sigma_gaussian_kernel(old_kernel_values: np.ndarray, old_sigma: float, new_sigma: float):

    # the gaussian kernel ist exp(-dist/(2*sigma^2)), if we want to change the sigma we do not need
    # to recompute since it is only an factor in an exponent.
    #
    # using the exponential rules:
    # (e^a)^b= e^(a*b), we can update the sigma using the following update rule
    # exp(-dist/(2*sigma_new^2)) = exp(-dist/(2*sigma_old^2))*(sigma_old^2/sigma_new^2)
    return old_kernel_values**((old_sigma**2)/(new_sigma**2))


@nb.njit()
def compute_H_hat(alpha=0.0, kernel_matrix_ref_samples=None, kernel_matrix_test_samples=None):
    """
    Calculates the H_hat term of the theta_hat optimization problem
    """
    n_ref = kernel_matrix_ref_samples.shape[1]
    n_test = kernel_matrix_test_samples.shape[1]

    H_hat = (alpha / n_ref) * kernel_matrix_ref_samples@kernel_matrix_ref_samples.T + \
            ((1.0 - alpha) / n_test) * kernel_matrix_test_samples@kernel_matrix_test_samples.T

    return H_hat


@nb.njit()
def compute_h_hat(kernel_matrix_ref_samples):
    """
    Calculates the h_hat term of the theta_hat optimization problem
    """
    h_hat = np.zeros((kernel_matrix_ref_samples.shape[0], 1))
    for idx in range(kernel_matrix_ref_samples.shape[0]):
        h_hat[idx, 0] = kernel_matrix_ref_samples[idx, :].mean()

    # unfortunately this is not numba friendly as axis parameter is not allowed
    # h_hat = np.mean(kernel_matrix_ref_samples, 1, keepdims=True)

    return h_hat


def compute_theta_hat(H_hat: np.ndarray, lambda_scaled_identity, h_hat: np.ndarray):
    """
    Calculates theta_hat given H_hat, h_hat, lambda, and the kernel basis function
    Treat as a system of linear equations and find the exact, optimal
    solution
    """
    # theta_hat = np.linalg.solve(H_hat + (lambda_regularizer * np.eye(kernel_basis)), h_hat)
    # due to the way H_hat is constructed it is symmetric (given some minor numerical errors)
    # see page 11 in [1] for the definition. As it is also real, its is automatically hermitian
    #
    # also we stole and shortened the scipy solve implementation to skip some Checks
    # theta_hat = scipy.linalg.solve(H_hat + (lambda_regularizer * np.eye(kernel_basis)), h_hat, assume_a='her',
    # check_finite=False)
    #
    # creating the identity within this function takes time (as we need to call the numpy function and create
    # the matrix) therefore we do it outside of the function
    return solve(H_hat + lambda_scaled_identity, h_hat)


@nb.njit()
def j_of_theta(alpha: float, g_xref_theta: np.ndarray, g_xtest_theta: np.ndarray):
    """
    Calculates the squared error criterion, J
    """
    return ((alpha / 2.0) * (np.mean(g_xref_theta ** 2)) +
            ((1 - alpha) / 2.0) * (np.mean(g_xtest_theta ** 2)) -
            np.mean(g_xref_theta))


@nb.njit()
def g_of_x_theta(kernel_matrix_samples: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """
    Calculate the alpha-relative density ratio kernel model
    """
    return theta_hat.T@kernel_matrix_samples


class AlphaRelativeDensityRatioEstimator:
    """
    Computes the alpha-relative density ratio estimate of P(X_ref) and P(X_test)
    The alpha-relative density ratio estimator, r_alpha(X), is given by the
    following kernel model:
    g(X; theta) = SUM( (theta_l * K(X, X_centers_l)), l=0, n )
    where theta is a vector of parameters [theta_1, theta_2, ..., theta_l]^T
    to be learned from the data samples. The parameters theta in the model
    g(X; theta) is calculated by solving the following optimization problem:
      theta_hat = argmin [ ( (1/2) * theta^T * H_hat * theta) - (h_hat^T * theta) + ( lambda/2 * theta^T * theta) ]
    where the expression (lambda/2 * theta^T * theta), with lambda >= 0, is
    a regularization term used to penalize against overfitting
    Reference:
    Relative Density-Ratio Estimation for Robust Distribution Comparison. Makoto Yamada,
    Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, and Masashi Sugiyama. NIPS,
    page 594-602. (2011)
    """

    def __init__(self, alpha_constraint=0.0, sigma_width=1.0, lambda_regularizer=0.0, kernel_basis=1):
        self.alpha_constraint = alpha_constraint
        self.sigma_width = sigma_width
        self.lambda_regularizer = lambda_regularizer
        self.kernel_basis = kernel_basis

    def apply(self, reference_samples: np.ndarray, test_samples: np.ndarray, gaussian_centers: np.ndarray) \
            -> (float, float):
        """
        Computes the alpha-relative density ratio, r_alpha(X), of P(X_ref) and P(X_test)
          r_alpha(X) = P(Xref) / (alpha * P(Xref) + (1 - alpha) * P(X_test)
        Returns density ratio estimate at X_ref, r_alpha_ref, and at X_test, r_alpha_test
        """
        # Apply the kernel function to the reference and test samples
        k_ref = compute_gaussian_kernel(reference_samples, gaussian_centers, self.sigma_width).T
        k_test = compute_gaussian_kernel(test_samples, gaussian_centers, self.sigma_width).T

        # Compute the parameters, theta_hat, of the density ratio estimator
        H_hat = compute_H_hat(self.alpha_constraint, k_ref, k_test)
        h_hat = compute_h_hat(k_ref)
        theta_hat = compute_theta_hat(H_hat, self.lambda_regularizer*np.eye(self.kernel_basis), h_hat)

        # Estimate the density ratio, r_alpha_ref = r_alpha(X_ref)
        r_alpha_ref = g_of_x_theta(k_ref, theta_hat)
        # Estimate the density ratio, r_alpha_test = r_alpha(X_test)
        r_alpha_test = g_of_x_theta(k_test, theta_hat)

        return r_alpha_ref, r_alpha_test


class PearsonRelativeDivergenceEstimator:
    """
    Calculates the alpha-relative Pearson divergence score
    The alpha-relative Pearson divergence is given by the following expression:
      PE_alpha = -(alpha/2(n_ref)) * SUM(r_alpha(X_ref_i)^2, i=0, n_ref)        -
                  ((1-alpha)/2(n_test)) * SUM(r_alpha(X_test_j)^2, j=0, n_test) +
                  (1/n_ref) * SUM(r_alpha(X_ref_i), i=0, n_ref)                 -
                  1/2
    where r_alpha(X) is the alpha-relative density ratio estimator and is given by
    the following kernel model:
      g(X; theta) = SUM( (theta_l * K(X, X_centers_l)), l=0, n )
    Reference:
    Relative Density-Ratio Estimation for Robust Distribution Comparison. Makoto
    Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, and Masashi Sugiyama.
    NIPS, page 594-602. (2011)
    """

    def __init__(self, alpha_constraint=0.0, sigma_width=1.0, lambda_regularizer=0.0, kernel_basis=1):
        self.alpha_constraint = alpha_constraint
        self.sigma_width = sigma_width
        self.lambda_regularizer = lambda_regularizer
        self.kernel_basis = kernel_basis

    def apply(self, reference_samples=None, test_samples=None, gaussian_centers=None):
        """
        Calculates the alpha-relative Pearson divergence score
        """
        density_ratio_estimator = AlphaRelativeDensityRatioEstimator(self.alpha_constraint, self.sigma_width,
                                                                     self.lambda_regularizer, self.kernel_basis)

        # Estimate alpha relative density ratio and pearson divergence score
        (r_alpha_Xref, r_alpha_Xtest) = density_ratio_estimator.apply(reference_samples, test_samples, gaussian_centers)

        pe_divergence = (np.mean(r_alpha_Xref) -
                         (0.5 * (self.alpha_constraint * np.mean(r_alpha_Xref ** 2) +
                                 (1.0 - self.alpha_constraint) * np.mean(r_alpha_Xtest ** 2))) - 0.5)

        return pe_divergence, r_alpha_Xtest


class RuLSIF:
    """
    Estimates the alpha-relative Pearson Divergence via the least Squares Relative
    Density Ratio Approximation
    Reference:
    Relative Density-Ratio Estimation for Robust Distribution Comparison. Makoto
    Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, and Masashi Sugiyama.
    NIPS, page 594-602. (2011)
    """

    def __init__(self, alpha=0.1, kernel_number=100, cross_folds=5, sigma: float = None, lambda_: float = None):
        self.alpha = alpha
        self.kernel_number = kernel_number
        self.cross_folds = cross_folds
        self.gaussian_centers = None
        self.sigma_width = sigma
        self.lambda_regularizer = lambda_

        # check if we need to set sigma or lambda using cross validation
        self.cv = self.sigma_width is None or self.lambda_regularizer is None

    @staticmethod
    def compute_gaussian_width_candidates(reference_samples: np.ndarray, test_samples: np.ndarray):
        """
        Compute a candidate list of Gaussian kernel widths. The best width will be
        selected via cross-validation

        Jaakkola's heuristic method for setting the width parameter of the Gaussian
        radial basis function kernel is to pick a quantile (usually the median) of
        the distribution of Euclidean distances between points having different
        labels.
        Reference:

        Jaakkola T S, Haussler D. Exploiting Generative Models in Discriminative Classifiers[J].
        Advances in Neural Information Processing Systems, 1998, 11(11):487--493.

        Jaakkola, M. Diekhaus, and D. Haussler. Using the Fisher kernel method to detect
        remote protein homologies. In T. Lengauer, R. Schneider, P. Bork, D. Brutlad, J.
        Glasgow, H.- W. Mewes, and R. Zimmer, editors, Proceedings of the Seventh
        International Conference on Intelligent Systems for Molecular Biology.

        It is the same technique as proposed in [1]
        """
        # create a complete sample array
        samples = np.c_[reference_samples, test_samples]

        # compute the pairwise distances
        distances = scipy.spatial.distance.pdist(samples.T, 'sqeuclidean')

        # apply the same formula as in
        # https://github.com/anewgithubname/change_detection/blob/main/lib/comp_med.m
        median_distance = np.sqrt(0.5 * np.median(distances[distances > 0]))

        return median_distance * np.array([0.6, 0.8, 1, 1.2, 1.4])

    @staticmethod
    def generate_regularization_params():
        """
        Generates a candidate list of regularization parameters to be used
        with the L1 regularizer term of RULSIF optimization problem.  The
        best regularizer parameter will be chosen via cross-validation.
        The values itself are taken from the paper [1].
        """
        return 10.0 ** np.array([-3, -2, -1, 0, 1])

    def generate_gaussian_centers(self, reference_samples=None):
        """
        Choose Gaussian centers randomly from the reference samples.
        """
        numcols = reference_samples.shape[1]
        reference_sample_idxs = np.random.permutation(numcols)

        self.kernel_number = min(self.kernel_number, numcols)
        gaussian_centers = reference_samples[:, reference_sample_idxs[0:self.kernel_number]]

        return gaussian_centers

    @staticmethod
    def cross_validate(reference_samples: np.ndarray, test_samples: np.ndarray, gaussian_centers: np.ndarray,
                       sigma_widths: np.ndarray, lambda_candidates: np.ndarray, alpha: float, kernel_number: int,
                       cross_folds=5):
        (refRows, refCols) = reference_samples.shape
        (testRows, testCols) = test_samples.shape

        # Initialize cross validation scoring matrix
        cross_validation_scores = np.zeros((sigma_widths.shape[0], lambda_candidates.shape[0]))

        # Initialize a cross validation index assignment list
        reference_samples_cv_idxs = np.random.permutation(refCols)
        reference_samples_cv_split = (np.arange(start=0, stop=refCols, step=1) * cross_folds) // refCols

        test_samples_cv_idxs = np.random.permutation(testCols)
        test_samples_cv_split = (np.arange(start=0, stop=testCols, step=1) * cross_folds) // testCols

        # initially calculate the kernel matrix using the candidate sigma width
        K_ref = compute_gaussian_kernel(reference_samples, gaussian_centers, sigma_widths[0]).T
        K_test = compute_gaussian_kernel(test_samples, gaussian_centers, sigma_widths[0]).T
        old_sigma = sigma_widths[0]

        # create an identity matrix which we need for computing theta
        # but as we can already create it here, we save time in our hottest path deep in the cross validation
        identity = np.eye(kernel_number)

        # Initiate k-fold cross-validation procedure. Using variable notation similar to the RuLSIF formulas in [1].
        for sigma_idx, sigma in enumerate(sigma_widths):

            # update the kernel for the new sigma without recomputing all distances
            K_ref = update_sigma_gaussian_kernel(K_ref, old_sigma, sigma)
            K_test = update_sigma_gaussian_kernel(K_test, old_sigma, sigma)

            # keep track of the sigma we used, so we keep updating our kernel
            old_sigma = sigma

            for fold_idx in range(cross_folds):

                # get the gaussian kernels for the current fold
                K_ref_trainingSet = K_ref[:, reference_samples_cv_idxs[reference_samples_cv_split != fold_idx]]
                K_test_trainingSet = K_test[:, test_samples_cv_idxs[test_samples_cv_split != fold_idx]]

                # compute the matrices for the current trainings fold
                H_h_KthFold = compute_H_hat(alpha, K_ref_trainingSet, K_test_trainingSet)
                h_h_KthFold = compute_h_hat(K_ref_trainingSet)

                # Select the subset of the kernel matrix not used in the training set
                # for use as the test set to validate against
                K_ref_testSet = K_ref[:, reference_samples_cv_idxs[reference_samples_cv_split == fold_idx]]
                K_test_testSet = K_test[:, test_samples_cv_idxs[test_samples_cv_split == fold_idx]]

                for lambda_idx, lambda_candidate in enumerate(lambda_candidates):
                    # Note: This is the absolute hot path of the complete function, if we can speed up anything
                    # within this loop, we will save a lot of time!

                    # compute the theta for the given parameters
                    theta_h_KthFold = compute_theta_hat(H_h_KthFold, identity*lambda_candidate, h_h_KthFold)

                    # compute the g_of_theta for the given parameters
                    r_alpha_Xref = g_of_x_theta(K_ref_testSet, theta_h_KthFold)
                    r_alpha_Xtest = g_of_x_theta(K_test_testSet, theta_h_KthFold)

                    # Calculate the objective function J(theta) under the current parameters and update the scores
                    cross_validation_scores[sigma_idx, lambda_idx] += j_of_theta(alpha, r_alpha_Xref, r_alpha_Xtest)

        return cross_validation_scores/cross_folds

    def compute_model_parameters(self, reference_samples=None, test_samples=None, gaussian_centers=None):
        """
        Computes model parameters via k-fold cross validation process
        """

        # get the parameter candidates
        sigma_widths = self.compute_gaussian_width_candidates(reference_samples, test_samples)
        lambda_candidates = self.generate_regularization_params()

        # compute the scores
        cross_validation_scores = self.cross_validate(reference_samples, test_samples, gaussian_centers, sigma_widths,
                                                      lambda_candidates, self.alpha, self.kernel_number,
                                                      self.cross_folds)

        # check for the minimal score within the cross_validation
        cv_min_idx_for_sigma, cv_min_idx_for_lambda = np.unravel_index(cross_validation_scores.argmin(),
                                                                       cross_validation_scores.shape)

        # get the optimal values for both parameters
        optimal_sigma = sigma_widths[cv_min_idx_for_sigma]
        optimal_lambda = lambda_candidates[cv_min_idx_for_lambda]
        return optimal_sigma, optimal_lambda

    def train(self, reference_samples=None, test_samples=None):
        """
        Learn the proper model parameters if we did not specify them already
        """

        self.gaussian_centers = self.generate_gaussian_centers(reference_samples)

        # check whether we need to find the model parameters or if they have been given
        if self.cv:
            optimal_sigma, optimal_lambda = self.compute_model_parameters(reference_samples,
                                                                          test_samples, self.gaussian_centers)

            self.sigma_width = optimal_sigma
            self.lambda_regularizer = optimal_lambda

    def apply(self, reference_samples=None, test_samples=None):
        """
        Estimates the alpha-relative Pearson divergence as determined by the relative
        ratio of probability densities:
           P(reference_samples[x]) / (alpha * P(reference_samples[x]) + (1 - alpha) * P(test_samples[x]))
        from samples:
           reference_samples[x_i] | reference_samples[x_i] in R^{d}, with i=1 to reference_samples{N}
        drawn independently of P(reference_samples[x])
        and from samples:
           test_samples[x_j] | test_samples[x_j] in R^{d}, with j=1 to test_samples{N}
        drawn independently from P(test_samples[x])
        After the model hyperparameters have been learned and chosen by the train()
        method, the RULSIF algorithm can be applied repeatedly on both in-sample and out
        of sample data
        """

        if self.gaussian_centers is None or self.kernel_number is None:
            raise Exception("Missing kernel basis function parameters")

        if self.sigma_width == 0.0 or self.lambda_regularizer == 0.0:
            raise Exception("Missing model selection parameters")

        divergence_estimator = PearsonRelativeDivergenceEstimator(self.alpha, self.sigma_width,
                                                                  self.lambda_regularizer, self.kernel_number)
        (pe_alpha, r_alpha_Xtest) = divergence_estimator.apply(reference_samples, test_samples, self.gaussian_centers)

        return pe_alpha

    def __call__(self, reference_samples: np.ndarray, test_samples: np.ndarray):

        # make the normalization as done in
        # https://github.com/anewgithubname/change_detection/blob/main/change_detection.m
        all_samples = np.c_[reference_samples, test_samples]
        std = np.std(all_samples, axis=1) + np.finfo(float).eps
        reference_samples /= std[:, None]
        test_samples /= std[:, None]

        # get the parameters estimation
        self.train(reference_samples, test_samples)

        # compute the result
        return self.apply(reference_samples, test_samples)


def simple_update_test():

    # make a random matrix of signals
    ref = np.random.normal(size=(10, 50))
    test = np.random.normal(size=(10, 50))

    # compute two gaussian kernels
    old_sigma = 1.1
    new_sigma = 3.4
    kernel_old = compute_gaussian_kernel(ref, test, old_sigma)
    kernel_new = compute_gaussian_kernel(ref, test, new_sigma)

    # compare with the kernel update function
    print(np.allclose(kernel_new, update_sigma_gaussian_kernel(kernel_old, old_sigma, new_sigma), atol=1.e-12))


if __name__ == '__main__':
    simple_update_test()
