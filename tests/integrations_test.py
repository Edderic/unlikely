"""
Module for integration testing.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..unlikely.engine import abc_smc
from ..unlikely.models import Models, Model
from ..unlikely.misc import create_images_from_data
from ..unlikely.priors import Beta

from ..conftest import assert_similar_enough_distribution


def test_beta_binomial_1():
    # A 1 is a "success", and a 0 is a "failure"
    obs = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1])

    # Number of particles to sample per epoch
    num_particles = 2000

    # The cutoff(s) that decide whether or not to accept a particle.
    epsilons = [3, 2, 1, 0]

    def distance(x, y):
        """
        For binomially distributed data, this essentially counts the number of
        "successes". We do that for both the observed and the simulated data
        sets and find the absolute distance between the two of them.

        This is for illustrative purposes only. You could write a more complex
        one that suits your own problem.

        Parameters:
            x: np.array
            y: np.array

        Returns: numeric
        """
        return abs(x.sum() - y.sum())

    def simulate(priors, actual_data):
        """
        Used by a model to simulate data.

        This is for illustrative purposes only. You could write a more complex
        one that suits your own problem.

        Parameters:
            priors: unlikely.priors.Priors
                 Acts like a dict. Keys should be names of priors of a model.

            actual_data: Some data

        Returns: integer
             A number between 0 and 1.
        """
        return np.random.binomial(
            n=1,
            p=priors['beta'],
            size=len(actual_data)
        )

    # Create a model. A model is a set of priors, plus a simulator
    models = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
        perturbation_param=0.9
    )

    # Compute the posterior distribution.
    abc_smc(
        num_particles,
        epsilons,
        models,
        np.array([obs[0]]),
        distance,
    )

    # Create a model that uses the full data set
    models_more_data = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
    )

    # Compute the posterior distribution for the models object with all the
    # data.
    abc_smc(
        num_particles,
        epsilons=epsilons,
        models=models_more_data,
        obs=obs,
        distance=distance,
    )

    # The posterior distribution (i.e. accepted particles that are compatible
    # "enough" with the data and model) are stored in
    # models[0].prev_accepted_proposals

    assert_similar_enough_distribution(
        models[0].prev_accepted_proposals,
        pd.DataFrame({'beta': np.random.beta(2, 1, num_particles)})
    )

    assert_similar_enough_distribution(
        models_more_data[0].prev_accepted_proposals,
        pd.DataFrame(
            {
                'beta': np.random.beta(
                    obs.sum() + 1, len(obs) - obs.sum() + 1,
                    num_particles
                )
            }
        )
    )

    # Assuming you have an "images" folder in your current working directory:
    create_images_from_data(
        save_path=Path(
            os.getenv("PWD")) / "images" / "beta_binomial_example.png",
        data={
            'title': "Comparison of Prior & Posterior of a Beta-Binomial",
            'data': [
                [
                    {
                        'title': 'Posterior after 1 success out of 1',
                        'data': [
                            models[0].prev_accepted_proposals.rename(
                                columns={'beta': 'posterior (eps: [3,2,1,0])'}
                            ),
                            pd.DataFrame(
                                {
                                    'reference_posterior': np.random.beta(
                                        2, 1, num_particles)
                                }
                            ),
                            pd.DataFrame(
                                {
                                    'prior': np.random.beta(
                                        1, 1, num_particles)
                                }
                            )
                        ]
                    },
                    {
                        'title': 'Full update with 6 successes out of 9',
                        'data': [
                            models_more_data[0].prev_accepted_proposals.rename(
                                columns={'beta': 'posterior (eps: [3,2,1,0])'}
                            ),
                            pd.DataFrame(
                                {
                                    'reference_posterior': np.random.beta(
                                        obs.sum() + 1,
                                        len(obs) - obs.sum() + 1,
                                        num_particles
                                    )
                                }
                            ),
                            pd.DataFrame(
                                {
                                    'prior': np.random.beta(
                                        1, 1, num_particles)
                                }
                            )
                        ]
                    }
                ]
            ]
        },
        xlim=(0, 1),
        figsize_mult=(5, 5)
    )
