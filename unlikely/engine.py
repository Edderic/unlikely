"""
Contains the abc_smc engine.
"""
import logging

from dask.distributed import as_completed
from tqdm import tqdm


def try_accepting_proposals(
    obs,
    epoch,
    models,
    distance,
    epsilon
):
    """
    Parallelizable part of ABC SMC.

    Parameters:
        model_proposer: ModelProposer
            Object that proposes what model to use, given the epoch.

        obs: the observed data

        epoch: integer
            The number representing what population we're trying to build up
            from the previous population.

        priors: list[Prior]
            The list of priors

        distance: callable
            The distance function we will be using to compare the simulated
            data vs. the observed

    Returns: tuple
        model: Model
            The model whose proposal we will accept
        perturbed_particle: dict
            The acceptable proposal by the model.
        sub_weights: dict
            keys: string
                corresponds to prior name.
            values: float
                The weights corresponding to each prior from the previous run.
    """

    dist = epsilon + 1
    while dist > epsilon:
        model = models.get_perturbed_proposal()
        perturbed_particle = model.get_perturbed_proposal(epoch)

        simulated_data = model.simulate(
            perturbed_particle,
            obs
        )

        dist = distance(
            simulated_data,
            obs
        )

    sub_weights = model.get_sub_weights(perturbed_particle, epoch)

    return model.get_name(), perturbed_particle, sub_weights


def abc_smc(  # pylint:disable=too-many-locals,too-many-arguments
    num_particles,
    epsilons,
    models,
    obs,
    distance,
    client=None,
):
    """
    Likelihood-free, Bayesian computation. This function runs the Approximate
    Bayesian Computation, Sequential Monte Carlo (ABC-SMC) of Tina Toni et al.

    Parameters:
        num_particles: integer
            The number of particles to sample for each epoch.

        epsilons: list[float]
            Distance limits that decide whether or not to accept a sample.

        models: Models
            A Models object.

        obs: data
            Observed data. Data structure could be anything.

        distance: callable
            Computes the distance between two data sets.

        client: dask.Distributed.Client
            If None, then we use sequential calculation of acceptable
            particles. Otherwise, this will run a parallelized search for
            acceptable particles.

    References:

    Tutorial on ABC rejection and ABC SMC for parameter estimation and model
    selection: https://arxiv.org/abs/0910.4472

    Approximate Bayesian computation scheme for parameter inference and model
    selection in dynamical systems.
    Tina Toni, David Welch, Natalja Strelkowa, Andreas Ipsen, Michael P.H.
    Stumpf: https://arxiv.org/abs/0901.1925

    https://stats.stackexchange.com/questions/326071/how-do-i-calculate-the-weights-in-abc-smc
    """
    logging.debug("num_particles: %s", num_particles)
    logging.debug("epsilons: %s", epsilons)
    logging.debug("obs: %s", obs)

    # epoch represents a population over time
    for epoch, epsilon in enumerate(epsilons):
        logging.debug("Current epoch and epsilon: %s, %s", epoch, epsilon)
        num_particles_accepted = 0

        if client is None:
            for _ in tqdm(range(num_particles)):
                model_name, proposal, weight = try_accepting_proposals(
                    obs,
                    epoch,
                    models,
                    distance,
                    epsilon
                )

                models.find_by_name(model_name).accept_proposal(
                    proposal, weight
                )

                num_particles_accepted += 1
                logging.debug(
                    "num_particles_accepted: %s",
                    num_particles_accepted
                )
        else:
            futures = []

            for _ in range(num_particles):
                futures.append(
                    client.submit(
                        try_accepting_proposals,
                        obs,
                        epoch,
                        models,
                        distance,
                        epsilon
                    )
                )

            for future in as_completed(futures):
                # Accepted
                model_name, proposal, sub_wt_particle = future.result()
                models.find_by_name(model_name).accept_proposal(
                    proposal, sub_wt_particle
                )

                num_particles_accepted += 1
                logging.debug(
                    "num_particles_accepted: %s",
                    num_particles_accepted
                )

                # Release from memory so garbage collection can take place.
                futures.remove(future)
                del future

        for model in models:
            model.cache_results()

        models.increment_num_epochs_processed()

        post_proba = models.get_posterior_probabilities()

        logging.debug(
            "epoch %s, models posterior proba: %s",
            models.num_epochs_processed,
            post_proba
        )
