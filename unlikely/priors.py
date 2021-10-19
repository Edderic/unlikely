"""
Classes for prior objects:

Continuous
----------
    Beta
    HalfCauchy
    Normal

DiscreteOrdinal
---------------
    TBD

DiscreteNonOrdinal
------------------
    TBD
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import beta as beta_dist, halfcauchy, norm, gaussian_kde, \
    uniform


class DistributionFromSamples(ABC):
    """
    Meant for batch processing / online Bayesian updating.

    Assuming that the samples are beta distributed, we recover

    Methods:
        - pdf
        - rvs
    """
    def __init__(self, samples, args=None):
        self.samples = samples
        self.kde = gaussian_kde(samples)
        if args is not None:
            self.args = args
        else:
            self.args = {}

    def rvs(self, size=None):
        """
        Randomly sample from this sampled distribution.
        """
        if size is None:
            return np.random.choice(self.samples)

        return np.random.choice(self.samples, size, replace=True)

    def pdf(self, val):
        """
        Uses Kernel Density Estimation (KDE) to estimate a probability
        distribution function (PDF).

        Parameters:
            val: float

        Returns: float
        """
        return self.kde(val)


class Prior(ABC):
    """
    Abstract Base Class for priors.

    Methods:
        compute_weight
        get_name

    Methods required for implementation  for subclasses:
        compute_weight
        perturb
        pdf
        sample
        __repr__
    """
    def __init__(self, distribution, name):
        self.distribution = distribution
        self.name = name
        self.constant_dev = None

    def compute_weight(
        self,
        particle,
        prev_weights,
        prev_values,
        prev_std,
    ):
        """
        Compute importance weight.

        Parameters:
            particle: float

            prev_weights: np.array
                1d array of floats.

            prev_values: np.array
                1d array of floats.

            prev_std: float
                Standard deviation of values from previous generation.
        """
        numerator = self.pdf(particle)

        if self.constant_dev is not None:
            std = self.constant_dev
        else:
            std = prev_std

        denominator = (
            prev_weights
            * Normal(
                prev_values,
                std
            ).pdf(particle)
        ).sum()

        return numerator / denominator

    def get_name(self):
        """
        Get the name of the prior.

        Returns: str
        """
        return self.name

    def use_constant_dev(self):
        """
        Set a constant deviation for all the perturbations throughout all
        epochs.

        For perturbation, use a constant standard deviation.
        """
        self.constant_dev = self.sample(10000).std()

    @abstractmethod
    def perturb(self, value, std):
        """
        Perturb the value using the standard deviation (std) parameter.

        Parameters:
            value: float
            std: float
                Greater than or equal to 0.

        Returns: float
        """

    def pdf(self, val):
        """
        Get the probability density function output for a value.

        Returns: float
        """
        return self.distribution.pdf(val)

    def sample(self, size=None):
        """
        Parameters:
            size: integer
                Greater than 0.

        Returns: float or np.array of length "size".
        """

        if size is None:
            return self.distribution.rvs()

        return self.distribution.rvs(size)

    # @abstractmethod
    # def use_sampling_distribution(self, samples):
        # self.

    @abstractmethod
    def __repr__(self):
        pass

    def use_distribution_from_samples(self, samples):
        """
        Creates a distribution out of samples.

        Parameters:
            samples: np.array
        """
        if self.constant_dev is not None:
            self.constant_dev = samples.std()


class BetaFromSamples(DistributionFromSamples):
    """
    Beta distribution from samples.
    """

    def __repr__(self):
        return "BetaFromSamples()"

    def pdf(self, val):
        """
        Uses Kernel Density Estimation (KDE) to estimate a probability
        distribution function (PDF).

        Parameters:
            val: float

        Returns: float
        """
        if val < 0 or val > 1:
            return 0

        return super().pdf(val)


class Beta(Prior):
    """
    Beta distribution.

    Good for modeling probabilities, since the range is constrained between 0
    and 1.
    """
    def __init__(
        self,
        alpha,
        beta,
        name=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.distribution = beta_dist(alpha, beta)
        self.distribution_from_samples_class = BetaFromSamples
        self.constant_dev = None
        Prior.__init__(self, self.distribution, name)

    def perturb(self, value, std):
        """
        This will perturb the particle and return something that is plausible.
        """

        counter = 0
        perturbation = -1

        if self.constant_dev is not None:
            standard_dev = self.constant_dev
        else:
            standard_dev = std

        while self.pdf(perturbation) == 0:
            if counter > 100:
                raise ValueError(
                    "Did not succeed producing viable perturbation."
                )

            perturbation = np.random.normal(value, standard_dev)
            counter += 1

        return perturbation

    def __repr__(self):
        return f"Beta(alpha: {self.alpha}, beta: {self.beta})"


class Uniform(Prior):
    """
    Uniform distribution.

    Creates a flat distribution between two numbers.
    """
    def __init__(
        self,
        alpha,
        beta,
        name=None,
    ):
        """
        Parameters:
            alpha: float
                Lower-bound
            beta: float
                Upper-bound
            name: string
        """
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.distribution = uniform(alpha, beta - alpha)
        self.distribution_from_samples_class = UniformFromSamples
        self.constant_dev = None
        Prior.__init__(self, self.distribution, name)

    def perturb(self, value, std):
        """
        This will perturb the particle and return something that is plausible.
        """

        counter = 0
        perturbation = -1

        if self.constant_dev is not None:
            standard_dev = self.constant_dev
        else:
            standard_dev = std

        while self.pdf(perturbation) == 0:
            if counter > 100:
                raise ValueError(
                    "Did not succeed producing viable perturbation."
                )

            perturbation = np.random.normal(value, standard_dev)
            counter += 1

        return perturbation

    def __repr__(self):
        return f"Beta(alpha: {self.alpha}, beta: {self.beta})"

    def use_distribution_from_samples(self, samples):
        """
        Creates a distribution out of samples.

        Parameters:
            samples: np.array
        """
        self.distribution = self.distribution_from_samples_class(
            samples,
            {
                'upper_bound': self.alpha,
                'lower_bound': self.beta
            }
        )

        super().use_distribution_from_samples(samples)


class UniformFromSamples(DistributionFromSamples):
    """
    Uniform Distribution from Samples
    """
    def __init__(self, samples, args):
        super(UniformFromSamples, self).__init__(samples, args)

    def __repr__(self):
        return "UniformFromSamples()"

    def pdf(self, val):
        """
        Uses Kernel Density Estimation (KDE) to estimate a probability
        distribution function (PDF).

        Parameters:
            val: float

        Returns: float
        """
        below_lower_bound = val < self.args['lower_bound']
        above_upper_bound = val > self.args['upper_bound']

        if below_lower_bound or above_upper_bound:
            return 0

        return super().pdf(val)


class NormalFromSamples(DistributionFromSamples):
    """
    Normal Distribution from Samples
    """


class Normal(Prior):
    """
    Normal Distribution
    """
    def __init__(self, mean, std, name=None):
        self.mean = mean
        self.std = std
        self.name = name
        self.distribution = norm(mean, std)
        self.distribution_from_samples_class = NormalFromSamples
        self.constant_dev = None
        Prior.__init__(self, self.distribution, name)

    def perturb(self, value, std):
        if self.constant_dev is not None:
            standard_dev = self.constant_dev
        else:
            standard_dev = std

        return np.random.normal(value, standard_dev)

    def __repr__(self):
        return f"Normal(mean: {self.mean}, std: {self.std})"


class HalfCauchyFromSamples(DistributionFromSamples):
    """
    HalfCauchyFromSamples
    """
    def pdf(self, val):
        """
        Half Cauchy does not produce values less than 0, so if we encounter
        that, return 0.
        """
        if val < 0:
            return 0.0

        return super().pdf(val)


class HalfCauchy(Prior):
    """
    Half Cauchy Distribution
    """
    def __init__(self, loc, scale, name):
        self.loc = loc
        self.scale = scale
        self.distribution = halfcauchy(loc, scale)
        self.name = name
        self.distribution_from_samples_class = HalfCauchyFromSamples
        self.constant_dev = None
        Prior.__init__(self, self.distribution, name)

    def perturb(self, value, std):
        """
        This will perturb the particle and return something that is plausible.
        """

        counter = 0
        perturbation = -1

        if self.constant_dev is not None:
            standard_dev = self.constant_dev
        else:
            standard_dev = std

        while self.pdf(perturbation) == 0:
            if counter > 100:
                raise ValueError(
                    "Did not succeed producing viable perturbation."
                )

            perturbation = np.random.normal(value, standard_dev)
            counter += 1

        return perturbation

    def __repr__(self):
        return f"HalfCauchy(loc: {self.loc}, scale: {self.scale})"
