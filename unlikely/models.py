"""
Models module.

Classes
-------
    Model
    Models
"""
import numpy as np
import pandas as pd


class Model():  # pylint:disable=too-many-instance-attributes
    """
    Model object has a bunch of "get_" methods to emphasize that these methods
    are not mutating state within the model instance. This lets us do
    parallelized simulations without the use of mutexes.

    Methods:
        accept_proposal
        get_accepted_count
        get_model_weight
        get_name
        get_particle_proposal
        get_perturbed_proposal
        get_prior_model_proba
        get_sub_weights
    """
    def __init__(  # pylint:disable=too-many-arguments
        self,
        name,
        priors,
        prior_model_proba,
        perturb=None,
        use_weighting_scheme=None,
        simulate=None
    ):
        """
        Parameters:
            name: string
            priors: list[Prior]
                The priors that we'll be sampling from, which the simulator
                uses.
            prior_model_proba: float
                Prior probability of this model being true.
            perturb: boolean. Defaults to True.
                If True, then model perturbs proposals. This is useful when you
                have a bunch of epsilons to go through, to prevent the
                posterior from getting too thin. The final posterior
                distribution can get too thin because the intermediary
                distributions can get too thin. They can get too thin because
                it's possible that we might not get enough samples from a
                distribution if our number of particles to collect is too
                small.

                However, there's a risk of overinflation, especially in the
                case of Bayesian updating with mini batches (i.e. updating
                iteratively with small batches of data).

                In those cases, perturbation, could inflate the variance,
                especially when the perturbation kernel gives proposals that
                are far from the original proposal, and there is not a lot of
                data within the batch. If this setting is set to False, it
                might be helpful to increase the number of particles to be
                collected, so that the posterior distribution doesn't get too
                thin (i.e. the posterior distribution coming out of this model
                doesn't have way smaller variance than the "true" posterior
                distribution.).

            use_weighting_scheme: boolean. Defaults to True. Experimental.
                If True, then each proposal essentially gets weighted by how
                far it is from the samples of the previous epoch, along with
                the weights of the previous epoch.

                The weighting scheme for a proposal is as follows:

                numerator = prior.pdf(proposal)
                denominator = sum_i prev_weight(prev_sample(i)) *
                    closeness_kernel.pdf(proposal, prev_sample(i))

                numerator / denominator gives us the weight.

                Intuitions:

                Decreasing the numerator leads to lower weight:

                Since the prior probability of a proposal is in the numerator,
                a low prior probability of a proposal will decrease the weight
                of the numerator.


                Decreasing the denominator leads to higher weight:

                If proposal is close to mostly low-weighted-samples, then the
                perturbation kernel pdf would give high values (since they are
                close to the proposal), but these will be dampened out by the
                low (normalized) weighting of said samples.


                Increasing the denominator leads to lower weight:

                If proposal is close to mostly highly-weighted samples, then
                the denominator will be large. High (normalized) weighting of
                the previous (accepted) proposals times their closeness to the
                sample leads to high values of the denominator. Big
                denominators shrink the assigned weight for the proposal.

                The weighting scheme, along with the perturbation parameter,
                prevents over-deflation of the posterior distribution. This is
                useful in the setting of having multiple epochs, where in each
                epoch, we are sampling from a finite sample.

            simulate: callable
                A callable that takes in the priors and produces data with it.
        """
        if perturb is None:
            self.perturb = True
        else:
            self.perturb = False

        if use_weighting_scheme is None:
            self.use_weighting_scheme = True
        else:
            self.use_weighting_scheme = False

        self.name = name
        self.num_epochs_processed = 0

        self.priors = {
            prior.get_name(): prior
            for prior in priors
        }

        self.prior_model_proba = prior_model_proba
        self.simulate_func = simulate

        self.accepted_proposals = [
            []
        ]

        self.weights = [
            []
        ]

        self.sub_weights = [
            []
        ]

        self.prev_accepted_proposals = None
        self.prev_stds = {
            prior.get_name(): prior.sample(1000).std()
            for prior in priors
        }

        self.prev_weights = None
        self.model_weights = [1]

    def accept_proposal(self, proposal, sub_weight, epoch=None):
        """
        Stores the proposal and the related sub_weights.

        Parameters:
            proposal: array
            sub_weight: array
            epoch: integer or None.
                If None, gets set to self.num_epochs_processed
        """
        if epoch is None:
            epoch = self.num_epochs_processed

        # prevent IndexError
        while epoch >= len(self.accepted_proposals):
            self.accepted_proposals.append([])
            self.sub_weights.append([])
            self.weights.append([])

        self.accepted_proposals[epoch].append(proposal)
        self.sub_weights[epoch].append(sub_weight)

    def get_accepted_count(self, epoch=None):
        """
        Get the number of accepted particles, given the epoch.

        Parameters:
            epoch: integer. Defaults to self.num_epochs_processed - 1
                Represents the population. An integer greater than or equal to
                0.

        Returns: integer
        """
        if epoch is None:
            epoch = self.num_epochs_processed - 1

        return len(self.accepted_proposals[epoch])

    def get_model_weight(self, epoch):
        """
        TODO: UNUSED. maybe remove.
        Get the weight of the model for a given epoch.

        Parameters:
            epoch: integer
                Represents the population. An integer greater than or equal to
                0.

        Returns: float
        """
        return self.model_weights[epoch]

    def get_name(self):
        """
        Return the name of the model.

        Returns: string
        """
        return self.name

    def get_particle_proposal(self, epoch):
        """
        Sample a particle.

        Sample from the priors if epoch is 0. Otherwise, sample from the
        population of the previous epoch.

        Parameters:
            epoch: integer
                Represents the population. An integer greater than or equal to
                0.

        """
        if epoch == 0:
            return {k: p.sample() for k, p in self.priors.items()}

        i = np.random.choice(
            list(range(len(self.accepted_proposals[epoch-1]))),
            p=self.weights[epoch-1]
        )

        return self.accepted_proposals[epoch-1][i]

    def get_perturbed_proposal(self, epoch=None):
        """
        Sample a perturbed particle.

        Sample from the priors if epoch is 0. Otherwise, sample from the
        population of the previous epoch.

        Parameters:
            epoch: integer
                Represents the population. An integer greater than or equal to
                0.

                Defaults to num_epochs_processed.

        Returns: dict
            Keys: strings
                Names of the corresponding priors.
            Values: Prior
        """
        if epoch is None:
            epoch = self.num_epochs_processed

        proposal = self.get_particle_proposal(epoch)

        if not self.perturb:
            return proposal

        perturbed = {
            prior_name: prior.perturb(
                proposal[prior_name],
                self.prev_stds[prior_name]
            )
            for prior_name, prior in zip(
                self.priors.keys(),
                self.priors.values(),
            )
        }

        return perturbed

    def get_prior_model_proba(self):
        """
        Gives the specified prior model probability.

        Returns: float
        """
        return self.prior_model_proba

    def get_sub_weights(self, perturbed, epoch=None):
        """
        Compute the weights of each subparticle, so that we can normalize each
        subparticle before multiplying them later into one weight for a
        particle (i.e. a set of subparticles). We normalize each subparticle
        before multiplying them later so that continuous distributions are not
        weighted more heavily than discrete distributions. Without
        normalization, this could happen because continuous distributions use
        probability density functions (pdfs) which can have values greater than
        1. In contrast, discrete distributions have probability mass functions
        (pmfs) which have an upper bound at 1.

        Parameters:
            perturbed: list
                This represents a particle, where each particle is a list of
                sub-particles.

            epoch: integer. Defaults to self.num_epochs_processed
                Represents a population of particles. Number greater than or
                equal to 0.

        Returns: dict
            key: string
                Corresponds to prior name
            values: float
                Weight corresponding to the prior name and epoch.
        """
        if epoch is None:
            epoch = self.num_epochs_processed

        prior_len = len(self.priors)

        if epoch == 0 or not self.use_weighting_scheme:
            return np.ones(prior_len)

        return {
            prior_name: prior.compute_weight(
                particle=perturbed[prior_name],
                prev_weights=self.prev_weights,
                prev_values=self.prev_accepted_proposals.loc[
                    :, prior_name
                ],
                prev_std=self.prev_stds[prior_name],
            )

            for (prior_name, prior) in self.priors.items()
        }

    def cache_results(self, epoch=None):
        """
        Do some caching of some variables so that the next iteration could use
        the previous epoch's data more easily.

        Parameters:
            epoch: integer. Defaults to self.num_epochs_processed
                Current epoch that will become the previous epoch

        Internally, we're storing data (e.g. accepted proposals) in a list
        that we can append to. It's not a fixed size array, which gives us
        flexibility. This flexibility is nice when we are simulating from more
        than one model (as there generally will be variations in acceptance
        probabilities of models). Instantiating fixed length numpy arrays from
        the beginning could technically work, but could be wasting memory, so
        instead, we:

        1. Maintain flexibility by keeping lists of accepted particles.

        2. Convert particles to fixed numpy arrays that we can apply numpy
        methods on, once the parallelized step of getting accepted particles
        are done. For example, functionality such as numpy slicing, e.g.
        some_np_array[1,:,3].

        Besides making it easier to compute results for the next epoch, this
        method also generates a single weight for each accepted particle.
        """

        if epoch is None:
            epoch = self.num_epochs_processed

        self.prev_accepted_proposals = pd.DataFrame(
            self.accepted_proposals[epoch]
        )

        self.prev_stds = self.prev_accepted_proposals.std(axis=0)
        # Each model can have a different set of priors, with the sets possibly
        # differing in length. We could multiply the weights of each prior or
        # sub-particles to get one number to represent the weight of the whole
        # particle. We can then use those combined weights to sample from the
        # model. However, priors could be of different types: discrete or
        # continuous. Continuous variables use pdf, while discrete variables
        # use pmf. The pdf can give larger values than 1, while the pmf is
        # upper bounded at 1. This could lead to the weights of continuous
        # variables being weighted more than discrete variables. To avoid
        # weighting continuous variables more than discrete, we normalize them
        # by variable type, which will scale the values to be less than or
        # equal to 1. Thus, we won't be overweighting continuous variables over
        # discrete.

        # NOTE:
        # Probably should do this in logspace since we're
        # multiplying a bunch of small numbers

        # sub_weights' length corresponds to the number of priors.
        sub_weights = pd.DataFrame(self.sub_weights[epoch])

        # We normalize each sub weight. This also has a length that corresponds
        # to the number of priors.
        normalized_tmp_weights = \
            sub_weights / sub_weights.sum(axis=0)

        # Multiply the sub_weights columns together to create one weight for
        # each particle.
        self.weights[epoch] = normalized_tmp_weights.prod(axis=1)
        self.model_weights.append(self.weights[epoch].sum())

        # Normalize the weights so that we can sample from it using
        # np.random.choice
        self.weights[epoch] = \
            self.weights[epoch] / self.weights[epoch].sum(axis=0)

        self.prev_weights = np.array(self.weights[epoch])

    def simulate(self, args):
        """
        Call the simulation function that was passed in.

        Parameters:
            args: dict
                Gets passed in to the simulation function.

        Returns: data
        """
        return self.simulate_func(self.priors, args)

    def increment_num_epochs_processed(self):
        """
        Useful for situations when we're updating the priors in a mini-batch
        fashion.
        """
        self.num_epochs_processed += 1

    def use_distribution_from_samples(self):
        """
        Uses the previously accepted proposals as new priors to sample from.
        """
        for prior_name, prior in self.priors.items():
            prior.use_distribution_from_samples(
                self.prev_accepted_proposals[prior_name]
            )

    def use_constant_dev(self):
        """
        For each prior, compute the standard deviation and then set it.
        """
        for _, prior in self.priors.items():
            prior.use_constant_dev()


class Models():
    """
    A collection of models.
    """
    def __init__(self, models, perturbation_param=None, use_constant_dev=None):
        """
        Parameters:
            models: unlikely.Models-like object
            perturbation_param: float. Defaults to 0.9
            use_constant_dev: boolean. Defaults to None
                If None or True, will delegate use_constant_dev method to the
                model.
        """
        self.models = models

        self.mapping = {}
        for model in models:
            name = model.get_name()
            if name in self.mapping.keys():
                raise ValueError("Names for models are not unique.")

            self.mapping[name] = model

        if perturbation_param is None:
            perturbation_param = 0.9

        self.perturbation_param = perturbation_param
        self.num_epochs_processed = 0

        if use_constant_dev is None or use_constant_dev:
            self.use_constant_dev()

    def __iter__(self):
        """
        Gives an instance the ability to be iterated. Iterates through the
        models. In other words, doing [m for m in models] where models is the
        Models instance, we would get instances of Model.
        """
        for model in self.models:
            yield model

    def __getitem__(self, index):
        return self.models[index]

    def find_by_name(self, name):
        """
        Parameters:
            name: string
        """
        return self.mapping[name]

    def get_models(self):
        """
        Returns: list[Model]
        """
        return self.models

    def get_posterior_probabilities(self, epoch=None):
        """
        Parameters:
            epoch: integer
                The integer that represents the population of interest.

        Returns: dict
            Keys are model names. Values are probabilities.
        """

        if epoch is None:
            epoch = self.num_epochs_processed

        weights = np.array([m.get_accepted_count() for m in self.models])
        probas = weights / weights.sum()

        return {
            m.get_name(): p for m, p in zip(self.models, probas)
        }

    def get_perturbed_proposal(self, epoch=None, perturb_proba=None):
        """
        Parameters:
            epoch: integer
                The integer that represents the population of interest.

            perturb_proba: float
                A perturbation probability. This will let us decide whether or
                not to randomly choose a model to try sampling from.
        Returns: Model
        """
        if epoch is None:
            epoch = self.num_epochs_processed

        if perturb_proba is None:
            perturb_proba = self.perturbation_param

        if np.random.binomial(n=1, p=perturb_proba):
            return self.get_proposal(epoch)

        # Uniformly pick any model
        probas = np.ones(len(self.get_models()))
        probas = probas / probas.sum()
        return np.random.choice(self.get_models(), p=probas)

    def get_proposal(self, epoch):
        """
        Parameters:
            epoch: integer
                The integer that represents the population of interest.

        Returns: Model
        """
        if epoch == 0:
            probas = [m.get_prior_model_proba() for m in self.models]
        else:
            # Might want to cache this for performance reasons.
            weights = np.array(
                [m.get_accepted_count() for m in self.models]
            )
            probas = weights / weights.sum()
        return np.random.choice(self.get_models(), p=probas)

    def increment_num_epochs_processed(self):
        """
        Useful for mini-batching.
        """
        self.num_epochs_processed += 1

        for model in self.models:
            model.increment_num_epochs_processed()

    def use_distribution_from_samples(self):
        """
        Tells each model to make use of distributions coming from data.
        """
        for model in self.models:
            model.use_distribution_from_samples()

    def use_constant_dev(self):
        """
        Tells each model to use constant standard deviation for perturbation
        and weighting.
        """
        for model in self.models:
            model.use_constant_dev()
