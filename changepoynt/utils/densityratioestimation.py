import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod

"""
Taken an modified from:
https://github.com/DeepOmics/rulsif_abrupt-change_detection
"""


class Estimator(ABC):
    """
    Marker interface for all Estimators. Provides basic apply() style
    interface and a show() method for debugging
    """
    @abstractmethod
    def apply(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def train(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def show(self):
        raise NotImplementedError


class GaussianKernel:

    def __init__(self, sigma=1.0):
        self.sigmaWidth = sigma

    def computeDistance(self, samples=None, sampleMeans=None):
        """
        Compute the distances between points in the sample's feature space
        to points along the center of the distribution
        """
        (sampleRows, sampleCols) = samples.shape
        (meanRows, meanCols) = sampleMeans.shape

        squaredSamples = np.sum(samples**2, 0)
        squaredMeans = np.sum(sampleMeans**2, 0)

        return np.tile(squaredMeans, (sampleCols, 1)) + np.tile(squaredSamples[:, None], (1, meanCols))\
            - 2 * np.dot(samples.T, sampleMeans)

    def apply(self, samples=None, sampleMeans=None):
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
        squaredDistance = self.computeDistance(samples, sampleMeans)
        return np.exp(-squaredDistance / (2 * (self.sigmaWidth**2)))


class AlphaRelativeDensityRatioEstimator(Estimator):
    """
    Computes the alpha-relative density ratio estimate of P(X_ref) and P(X_test)
    The alpha-relative density ratio estimator, r_alpha(X), is given by the
    following kernel model:
    g(X; theta) = SUM( (theta_l * K(X, X_centers_l)), l=0, n )
    where theta is a vector of parameters [theta_1, theta_2, ..., theta_l]^T
    to be learned from the data samples. The parameters theta in the model
    g(X; theta) is calculated by solving the following optimization problem:
      theta_hat = argmin [ ( (1/2) * theta^T * H_hat * theta) - (h_hat^T * theta) + ( lambda/2 * theta^T * theta) ]
    where the expression (lamba/2 * theta^T * theta), with lambda >= 0, is
    a regularization term used to penalize against overfitting
    Reference:
    Relative Density-Ratio Estimation for Robust Distribution Comparison. Makoto Yamada,
    Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, and Masashi Sugiyama. NIPS,
    page 594-602. (2011)
    """
    alphaConstraint = None
    sigmaWidth = None
    lambdaRegularizer = None
    kernelBasis = None

    def __init__(self, alphaConstraint=0.0, sigmaWidth=1.0, lambdaRegularizer=0.0, kernelBasis=1):
        self.alphaConstraint = alphaConstraint
        self.sigmaWidth = sigmaWidth
        self.lambdaRegularizer = lambdaRegularizer
        self.kernelBasis = kernelBasis

    def train(self, x, y):
        pass

    def show(self):
        pass

    def apply(self, referenceSamples=None, testSamples=None, gaussianCenters=None):
        """
        Computes the alpha-relative density ratio, r_alpha(X), of P(X_ref) and P(X_test)
          r_alpha(X) = P(Xref) / (alpha * P(Xref) + (1 - alpha) * P(X_test)
        Returns density ratio estimate at X_ref, r_alpha_ref, and at X_test, r_alpha_test
        """
        # Apply the kernel function to the reference and test samples
        K_ref = GaussianKernel(self.sigmaWidth).apply(referenceSamples, gaussianCenters).T
        K_test = GaussianKernel(self.sigmaWidth).apply(testSamples, gaussianCenters).T

        # Compute the parameters, theta_hat, of the density ratio estimator
        H_hat = AlphaRelativeDensityRatioEstimator.H_hat(self.alphaConstraint, K_ref, K_test)
        h_hat = AlphaRelativeDensityRatioEstimator.h_hat(K_ref)
        theta_hat = AlphaRelativeDensityRatioEstimator.theta_hat(H_hat, h_hat, self.lambdaRegularizer, self.kernelBasis)

        # Estimate the density ratio, r_alpha_ref = r_alpha(X_ref)
        r_alpha_ref = AlphaRelativeDensityRatioEstimator.g_of_X_theta(K_ref, theta_hat).T
        # Estimate the density ratio, r_alpha_test = r_alpha(X_test)
        r_alpha_test = AlphaRelativeDensityRatioEstimator.g_of_X_theta(K_test, theta_hat).T

        # print(r_alpha_ref)
        # print(r_alpha_test)

        # distance_type = ['Hellinger Distance',]

        # HD  = abs(r_alpha_ref- r_alpha_test)

        return r_alpha_ref, r_alpha_test

    @staticmethod
    def H_hat(alpha=0.0, KernelMatrix_refSamples=None, KernelMatrix_testSamples=None):
        """
        Calculates the H_hat term of the theta_hat optimization problem
        """
        N_ref = KernelMatrix_refSamples.shape[1]
        N_test = KernelMatrix_testSamples.shape[1]

        H_hat = (alpha / N_ref) * np.dot(KernelMatrix_refSamples, KernelMatrix_refSamples.T) + \
                ((1.0 - alpha) / N_test) * np.dot(KernelMatrix_testSamples, KernelMatrix_testSamples.T)

        return H_hat

    @staticmethod
    def h_hat(KernelMatrix_refSamples):
        """
        Calculates the h_hat term of the theta_hat optimization problem
        """
        h_hat = np.mean(KernelMatrix_refSamples, 1)

        return h_hat

    @staticmethod
    def theta_hat(H_hat=None, h_hat=None, lambdaRegularizer=0.0, kernelBasis=None):
        """
        Calculates theta_hat given H_hat, h_hat, lambda, and the kernel basis function
        Treat as a system of linear equations and find the exact, optimal solution
        """
        theta_hat = linalg.solve(H_hat + (lambdaRegularizer * np.eye(kernelBasis)), h_hat)

        return theta_hat

    @staticmethod
    def J_of_theta(alpha=0.0, g_Xref_theta=None, g_Xtest_theta=None):
        """
        Calculates the squared error criterion, J
        """
        return ((alpha / 2.0) * (np.mean(g_Xref_theta ** 2)) +
                ((1 - alpha) / 2.0) * (np.mean(g_Xtest_theta ** 2)) -
                np.mean(g_Xref_theta))

    @staticmethod
    def g_of_X_theta(KernelMatrix_samples=None, theta_hat=None):
        """
        Calculate the alpha-relative density ratio kernel model
        """
        return np.dot(KernelMatrix_samples.T, theta_hat)


class PearsonRelativeDivergenceEstimator(Estimator):
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
    alphaConstraint = None
    sigmaWidth = None
    lambdaRegularizer = None
    kernelBasis = None

    def __init__(self, alphaConstraint=0.0, sigmaWidth=1.0, lambdaRegularizer=0.0, kernelBasis=1):
        self.alphaConstraint = alphaConstraint
        self.sigmaWidth = sigmaWidth
        self.lambdaRegularizer = lambdaRegularizer
        self.kernelBasis = kernelBasis

    def train(self, x, y):
        pass

    def show(self):
        pass

    def apply(self, referenceSamples=None, testSamples=None, gaussianCenters=None):
        """
        Calculates the alpha-relative Pearson divergence score
        """
        densityRatioEstimator = AlphaRelativeDensityRatioEstimator(self.alphaConstraint,
                                                                   self.sigmaWidth,
                                                                   self.lambdaRegularizer,
                                                                   self.kernelBasis)

        # Estimate alpha relative density ratio and pearson divergence score
        (r_alpha_Xref, r_alpha_Xtest) = densityRatioEstimator.apply(referenceSamples, testSamples, gaussianCenters)

        PE_divergence = (np.mean(r_alpha_Xref) -
                         (0.5 * (self.alphaConstraint * np.mean(r_alpha_Xref ** 2) +
                                 (1.0 - self.alphaConstraint) * np.mean(r_alpha_Xtest ** 2))) - 0.5)

        return PE_divergence, r_alpha_Xtest


class RULSIF(Estimator):
    """
    Estimates the alpha-relative Pearson Divergence via Least Squares Relative
    Density Ratio Approximation
    Reference:
    Relative Density-Ratio Estimation for Robust Distribution Comparison. Makoto
    Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, and Masashi Sugiyama.
    NIPS, page 594-602. (2011)
    """

    def __init__(self, alpha: float = 0.0, sigma: float = None, lambda_regularization: float = None,
                 n_kernels: int = 100, cross_folds: int = 5, verbose: bool = False):

        self.alphaConstraint = alpha
        self.sigmaWidth = sigma
        self.lambdaRegularizer = lambda_regularization
        self.kernelBasis = n_kernels
        self.crossFolds = cross_folds
        self.verbose = verbose

    def getMedianDistanceBetweenSamples(self, sampleSet=None):
        """
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
        """
        numrows = sampleSet.shape[0]
        samples = sampleSet

        G = np.sum((samples * samples), axis=1)
        Q = np.tile(G[:, None], (1, numrows))
        R = np.tile(G, (numrows, 1))

        distances = Q + R - 2 * np.dot(samples, samples.T)

        distances = distances - np.tril(distances)
        distances = distances.reshape(numrows ** 2, 1, order="F").copy()

        return np.sqrt(0.5 * np.median(distances[distances > 0]))

    def computeGaussianWidthCandidates(self, referenceSamples=None, testSamples=None):
        """
        Compute a candidate list of Gaussian kernel widths. The best width will be
        selected via cross-validation
        """
        allSamples = np.c_[referenceSamples, testSamples]
        medianDistance = self.getMedianDistanceBetweenSamples(allSamples.T)

        return medianDistance * np.array([0.6, 0.8, 1, 1.2, 1.4])

    def generateRegularizationParams(self):
        """
        Generatees a candidate list of regularization parameters to be used
        with the L1 regularizer term of RULSIF optimization problem.  The
        best regularizer parameter will be chosen via cross-validation
        """
        return 10.0 ** np.array([-3, -2, -1, 0, 1])

    def generateAllGaussianCenters(self, referenceSamples=None):
        """
        Generates kernels in the region where the P(X_reference) takes large values
        """
        self.kernelBasis = referenceSamples.shape[1]
        return referenceSamples[:, np.r_[0:self.kernelBasis]]

    def generateRandomGaussianCenters(self, referenceSamples=None):
        """
        Randomly chooses Gaussian centers as an optimization
        """
        numcols = referenceSamples.shape[1]
        referenceSampleIdxs = np.random.permutation(numcols)

        self.kernelBasis = min(self.kernelBasis, numcols)
        return referenceSamples[:, referenceSampleIdxs[0:self.kernelBasis]]

    def generateFirstNGaussianCenters(self, referenceSamples=None):
        """
        Chooses the firts N samples as Gaussian centers as an optimization
        """
        numcols = referenceSamples.shape[1]
        self.kernelBasis = min(self.kernelBasis, numcols)
        return referenceSamples[:, np.r_[0:self.kernelBasis]]

    def generateGaussianCenters(self, referenceSamples=None):
        """
        Choose Gaussian centers based on a strategy
        """
        gaussianCenters = self.generateAllGaussianCenters(referenceSamples)

        return gaussianCenters

    def computeModelParameters(self, referenceSamples=None, testSamples=None, gaussianCenters=None):
        """
        Computes model parameters via k-fold cross validation process
        """
        (refRows, refCols) = referenceSamples.shape
        (testRows, testCols) = testSamples.shape

        sigmaWidths = self.computeGaussianWidthCandidates(referenceSamples, testSamples)
        lambdaCandidates = self.generateRegularizationParams()

        # Initialize cross validation scoring matrix
        crossValidationScores = np.zeros((np.size(sigmaWidths), np.size(lambdaCandidates)))

        # Initialize a cross validation index assignment list
        referenceSamplesCVIdxs = np.random.permutation(refCols)
        referenceSamplesCVSplit = np.floor(np.r_[0:refCols] * self.crossFolds / refCols)
        testSamplesCVIdxs = np.random.permutation(testCols)
        testSamplesCVSplit = np.floor(np.r_[0:testCols] * self.crossFolds / testCols)

        # Initiate k-fold cross-validation procedure. Using variable
        # notation similar to the RULSIF formulas.
        for sigmaIdx in np.r_[0:np.size(sigmaWidths)]:

            # (re-)Calculate the kernel matrix using the candidate sigma width
            sigma = sigmaWidths[sigmaIdx]
            K_ref = GaussianKernel(sigma).apply(referenceSamples, gaussianCenters).T
            K_test = GaussianKernel(sigma).apply(testSamples, gaussianCenters).T

            # Initialize a new result matrix for the current sigma candidate
            foldResult = np.zeros((self.crossFolds, np.size(lambdaCandidates)))

            for foldIdx in np.r_[0:self.crossFolds]:

                K_ref_trainingSet = K_ref[:, referenceSamplesCVIdxs[referenceSamplesCVSplit != foldIdx]]
                K_test_trainingSet = K_test[:, testSamplesCVIdxs[testSamplesCVSplit != foldIdx]]
                H_h_KthFold = AlphaRelativeDensityRatioEstimator.H_hat(self.alphaConstraint, K_ref_trainingSet,
                                                                       K_test_trainingSet)
                h_h_KthFold = AlphaRelativeDensityRatioEstimator.h_hat(K_ref_trainingSet)

                for lambdaIdx in np.r_[0:np.size(lambdaCandidates)]:
                    lambdaCandidate = lambdaCandidates[lambdaIdx]

                    theta_h_KthFold = AlphaRelativeDensityRatioEstimator.theta_hat(H_h_KthFold, h_h_KthFold,
                                                                                   lambdaCandidate, self.kernelBasis)

                    # Select the subset of the kernel matrix not used in the training set
                    # for use as the test set to validate against
                    K_ref_testSet = K_ref[:, referenceSamplesCVIdxs[referenceSamplesCVSplit == foldIdx]]
                    K_test_testSet = K_test[:, testSamplesCVIdxs[testSamplesCVSplit == foldIdx]]

                    r_alpha_Xref = AlphaRelativeDensityRatioEstimator.g_of_X_theta(K_ref_testSet, theta_h_KthFold)
                    r_alpha_Xtest = AlphaRelativeDensityRatioEstimator.g_of_X_theta(K_test_testSet, theta_h_KthFold)

                    # Calculate the objective function J(theta) under the current parameters
                    J = AlphaRelativeDensityRatioEstimator.J_of_theta(self.alphaConstraint, r_alpha_Xref, r_alpha_Xtest)

                    foldResult[foldIdx, lambdaIdx] = J

                crossValidationScores[sigmaIdx, :] = np.mean(foldResult, 0)

        crossValidationMinScores = np.min(crossValidationScores, axis=1)
        crossValidationMinIdxForLambda = crossValidationScores.argmin(1)
        crossValidationMinIdxForSigma = crossValidationMinScores.argmin()

        optimalSigma = sigmaWidths[crossValidationMinIdxForSigma]
        optimalLambda = lambdaCandidates[crossValidationMinIdxForLambda[crossValidationMinIdxForSigma]]

        return (optimalSigma, optimalLambda)

    def train(self, referenceSamples=None, testSamples=None):
        """
        Learn the proper model parameters
        """

        self.gaussianCenters = self.generateGaussianCenters(referenceSamples)

        (optimalSigma, optimalLambda) = self.computeModelParameters(referenceSamples, testSamples, self.gaussianCenters)

        self.sigmaWidth = optimalSigma
        self.lambdaRegularizer = optimalLambda

    def apply(self, referenceSamples=None, testSamples=None):
        """
        Estimates the alpha-relative Pearson divergence as determined by the relative
        ratio of probability densities:
           P(ReferenceSamples[x]) / (alpha * P(ReferenceSamples[x]) + (1 - alpha) * P(TestSamples[x]))
        from samples:
           ReferenceSamples[x_i] | ReferenceSamples[x_i] in R^{d}, with i=1 to ReferenceSamples{N}
        drawn independently from P(ReferenceSamples[x])
        and from samples:
           TestSamples[x_j] | TestSamples[x_j] in R^{d}, with j=1 to TestSamples{N}
        drawn independently from P(TestSamples[x])
        After the model hyper-parameters have been learned and chosen by the train()
        method, the RULSIF algorithm can be applied repeatedly on both in-sample and out
        of sample data
        """

        if self.gaussianCenters is None or self.kernelBasis is None:
            raise Exception("Missing kernel basis function parameters")

        if self.sigmaWidth == 0.0 or self.lambdaRegularizer == 0.0:
            raise Exception("Missing model selection parameters")

        divergenceEstimator = PearsonRelativeDivergenceEstimator(self.alphaConstraint, self.sigmaWidth,
                                                                 self.lambdaRegularizer, self.kernelBasis)
        (PE_alpha, r_alpha_Xtest) = divergenceEstimator.apply(referenceSamples, testSamples, self.gaussianCenters)

        self.show("RULSIF Results", self.sigmaWidth, self.lambdaRegularizer, PE_alpha, self.verbose)

        return PE_alpha

    def show(self, displayName=None, optimalSigma=0.0, optimalLambda=0.0, PE_alpha=0.0, verbose=False):
        if not verbose:
            return

        print("[" + displayName + "]\n")
        print("Alpha Constraint         : " + str(self.alphaConstraint))
        print("Kernel Basis Functions   : " + str(self.kernelBasis))
        print("Basis Function Width     : " + str(optimalSigma))
        print("Regularization Parameter : " + str(optimalLambda))
        print("Pearson Divergence Score : " + str(PE_alpha))
        print("\n")
        print("---------------")