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
from scipy.stats import beta as beta_dist, halfcauchy, norm, gaussian_kde

from .misc import find_index_of_closest

class DistributionFromSamples(ABC):
    """
    Meant for batch processing / online Bayesian updating.

    Assuming that the samples are beta distributed, we recover

    Methods:
        - pdf
        - rvs
    """
    def __init__(self, samples):
        self.samples = samples
        self.kde = gaussian_kde(samples)

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
    def __init__(self, distribution):
        self.distribution = distribution

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

        denominator = (
            prev_weights
            * Normal(
                prev_values,
                prev_std
            ).pdf(particle)
        ).sum()

        return numerator / denominator

    def get_name(self):
        return self.name

    def use_distribution_from_samples(self, samples):
        """
        Creates a distribution out of samples.

        Parameters:
            samples: np.array
        """
        self.distribution = self.distribution_from_samples_class(samples)

    @abstractmethod
    def perturb(self, value, std):
        pass

    def pdf(self, val):
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

class BetaFromSamples(DistributionFromSamples):
    def __repr__(self):
        return f"BetaFromSamples()"

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
    def __init__(self, alpha, beta, name=None):
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.distribution = beta_dist(alpha, beta)
        self.distribution_from_samples_class = BetaFromSamples

    def perturb(self, value, std):
        """
        This will perturb the particle and return something that is plausible.
        """

        counter = 0
        perturbation = -1

        while self.pdf(perturbation) == 0:
            if counter > 100:
                raise ValueError(
                    f"Did not succeed producing viable perturbation."
                )

            perturbation = np.random.normal(value, std)
            counter += 1

        return perturbation

    def __repr__(self):
        return f"Beta(alpha: {self.alpha}, beta: {self.beta})"

class NormalFromSamples(DistributionFromSamples):
    pass

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

    def perturb(self, value, std):
        return np.random.normal(value, std)

    def __repr__(self):
        return f"Normal(mean: {self.loc}, std: {self.scale})"

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

    def perturb(self, value, std):
        """
        This will perturb the particle and return something that is plausible.
        """

        counter = 0
        perturbation = -1

        while self.pdf(perturbation) == 0:
            if counter > 100:
                raise ValueError(f"Did not succeed producing viable perturbation.")

            perturbation = np.random.normal(value, std)
            counter += 1

        return perturbation

    def __repr__(self):
        return f"HalfCauchy(loc: {self.loc}, scale: {self.scale})"
