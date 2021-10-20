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
from ..unlikely.priors import Beta, Uniform

from .conftest import assert_similar_enough_distribution


def test_beta_binomial_1():
    # A 1 is a "success", and a 0 is a "failure"
    obs = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1])

    # Number of particles to sample per epoch
    num_particles = 2000

    # The cutoff(s) that decide whether or not to accept a particle.
    epsilon_sets = [[0], [1, 0], [3, 2, 1, 0]]
    column = [
        {
            'title': 'Posterior after 1 success out of 1',
            'data': [
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

    models_list = []

    for i, epsilons in enumerate(epsilon_sets):
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

        models_list.append(models)

        # Compute the posterior distribution.
        abc_smc(
            num_particles,
            epsilons,
            models,
            np.array([obs[0]]),
            distance,
        )

        column[0]['data'].append(
            models[0].prev_accepted_proposals.rename(
                columns={'beta': f'eps: {epsilons}'}
            )
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

        models_list.append(models_more_data)
        # Compute the posterior distribution for the models object with all the
        # data.
        abc_smc(
            num_particles,
            epsilons=epsilons,
            models=models_more_data,
            obs=obs,
            distance=distance,
        )

        column[1]['data'].append(
            models_more_data[0].prev_accepted_proposals.rename(
                columns={'beta': f'eps: {epsilons}'}
            )
        )
    # The posterior distribution (i.e. accepted particles that are compatible
    # "enough" with the data and model) are stored in
    # models[0].prev_accepted_proposals

    # Assuming you have an "images" folder in your current working directory:
    create_images_from_data(
        save_path=Path(
            os.getenv("PWD")) / "images" / "beta_binomial_example.png",
        data={
            'title': "Comparison of Prior & Posterior of a Beta-Binomial",
            'data': [
               column
            ]
        },
        xlim=(0, 1),
        figsize_mult=(5, 5)
    )

    for i in range(len(models_list)):
        if i % 2 == 0:
            assert_similar_enough_distribution(
                models_list[i][0].prev_accepted_proposals,
                pd.DataFrame({'beta': np.random.beta(2, 1, num_particles)})
            )

        else:
            assert_similar_enough_distribution(
                models_list[i][0].prev_accepted_proposals,
                pd.DataFrame(
                    {
                        'beta': np.random.beta(
                            obs.sum() + 1, len(obs) - obs.sum() + 1,
                            num_particles
                        )
                    }
                )
            )


def test_uniform_binomial_1():
    # A 1 is a "success", and a 0 is a "failure"
    obs = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1])

    # Number of particles to sample per epoch
    num_particles = 2000

    # The cutoff(s) that decide whether or not to accept a particle.
    epsilon_sets = [[0], [1, 0], [3, 2, 1, 0]]
    column = [
        {
            'title': 'Posterior after 1 success out of 1',
            'data': [
                pd.DataFrame(
                    {
                        'reference_posterior': np.random.beta(
                            2, 1, num_particles)
                    }
                ),
                pd.DataFrame(
                    {
                        'prior': np.random.uniform(
                            0, 1, num_particles)
                    }
                )
            ]
        },
        {
            'title': 'Full update with 6 successes out of 9',
            'data': [
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
                        'prior': np.random.uniform(
                            0, 1, num_particles)
                    }
                )
            ]
        }
    ]

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
            p=priors['uniform'],
            size=len(actual_data)
        )

    models_list = []

    for i, epsilons in enumerate(epsilon_sets):
        # Create a model. A model is a set of priors, plus a simulator
        models = Models(
            [
                Model(
                    name='flat prior',
                    priors=[
                        Uniform(alpha=0, beta=1, name="uniform"),
                    ],
                    simulate=simulate,
                    prior_model_proba=1
                ),
            ],
            perturbation_param=0.9
        )

        models_list.append(models)

        # Compute the posterior distribution.
        abc_smc(
            num_particles,
            epsilons,
            models,
            np.array([obs[0]]),
            distance,
        )

        column[0]['data'].append(
            models[0].prev_accepted_proposals.rename(
                columns={'uniform': f'eps: {epsilons}'}
            )
        )

        # Create a model that uses the full data set
        models_more_data = Models(
            [
                Model(
                    name='flat prior',
                    priors=[
                        Uniform(alpha=0, beta=1, name="uniform"),
                    ],
                    simulate=simulate,
                    prior_model_proba=1
                ),
            ],
        )

        models_list.append(models_more_data)
        # Compute the posterior distribution for the models object with all the
        # data.
        abc_smc(
            num_particles,
            epsilons=epsilons,
            models=models_more_data,
            obs=obs,
            distance=distance,
        )

        column[1]['data'].append(
            models_more_data[0].prev_accepted_proposals.rename(
                columns={'uniform': f'eps: {epsilons}'}
            )
        )
    # The posterior distribution (i.e. accepted particles that are compatible
    # "enough" with the data and model) are stored in
    # models[0].prev_accepted_proposals

    # Assuming you have an "images" folder in your current working directory:
    create_images_from_data(
        save_path=Path(
            os.getenv("PWD")) / "images" / "uniform_binomial_example.png",
        data={
            'title': "Comparison of Prior & Posterior of a Uniform-Binomial",
            'data': [
               column
            ]
        },
        xlim=(0, 1),
        figsize_mult=(5, 5)
    )

    for i in range(len(models_list)):
        if i % 2 == 0:
            assert_similar_enough_distribution(
                models_list[i][0].prev_accepted_proposals,
                pd.DataFrame({'uniform': np.random.beta(2, 1, num_particles)})
            )

        else:
            assert_similar_enough_distribution(
                models_list[i][0].prev_accepted_proposals,
                pd.DataFrame(
                    {
                        'uniform': np.random.beta(
                            obs.sum() + 1, len(obs) - obs.sum() + 1,
                            num_particles
                        )
                    }
                )
            )


def test_uniform_binomial_2():
    num_particles = 2000
    obs = np.array([
        1, 1, 1,
        1, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 1
    ])

    epsilons_list = [[0], [3, 2, 1, 0]]

    def distance(x, y):
        """
        Compare the number of ones in one vs. the other.
        """
        return abs(x.sum() - y.sum())

    def simulate(priors, obs):
        """
        Data is binomially distributed.
        """
        return np.random.binomial(n=1, p=priors['uniform'], size=len(obs))

    data_to_display = [
        [
            {
                'title': f"obs: {obs[:3]}",
                'data': []
            },
            {
                'title': f"after {obs[3:7]}",
                'data': []
            },
            {
                'title': f"after {obs[7:]}",
                'data': []
            },
            {
                'title': "Full batch",
                'data': []
            }
        ]
    ]

    for row, epsilons in enumerate(epsilons_list):
        models = Models(
            [
                Model(
                    name='Uniform over (0.5, 1)',
                    priors=[
                        Uniform(alpha=0.5, beta=1, name="uniform"),
                    ],
                    simulate=simulate,
                    prior_model_proba=1,
                ),
            ]
        )

        # Update with 1st batch
        abc_smc(
            num_particles=num_particles,
            epsilons=epsilons,
            models=models,
            obs=obs[:3],
            distance=distance,
        )

        data_to_display[0][0]['data'].append(
            pd.DataFrame(models[0].prev_accepted_proposals).rename(
                columns={'uniform': f'eps: {epsilons}'}
            )
        )

        # The posterior distribution becomes the prior
        models.use_distribution_from_samples()

        # Update with 2nd batch
        abc_smc(
            num_particles=num_particles,
            epsilons=epsilons,
            models=models,
            obs=obs[3:7],
            distance=distance,
        )

        # The posterior distribution becomes the prior
        models.use_distribution_from_samples()

        data_to_display[0][1]['data'].append(
            pd.DataFrame(
                models[0].prev_accepted_proposals).rename(
                columns={'uniform': f'eps: {epsilons}'}
            )
        )

        # Update with 3rd batch
        abc_smc(
            num_particles=num_particles,
            epsilons=epsilons,
            models=models,
            obs=obs[7:],
            distance=distance,
        )

        data_to_display[0][2]['data'].append(
            pd.DataFrame(
                models[0].prev_accepted_proposals).rename(
                columns={'uniform': f'eps: {epsilons}'}
            )
        )

        models_full_batch = Models(
            [
                Model(
                    name='flat prior',
                    priors=[
                        Uniform(alpha=0.5, beta=1, name="uniform"),
                    ],
                    simulate=simulate,
                    prior_model_proba=1,
                ),
            ]
        )

        # Update full batch
        abc_smc(
            num_particles=num_particles,
            epsilons=epsilons,
            models=models_full_batch,
            obs=obs,
            distance=distance,
        )

        data_to_display[0][3]['data'].append(
            pd.DataFrame(
                models_full_batch[0].prev_accepted_proposals
            ).rename(columns={'uniform': f'eps: {epsilons}'})
        )

    create_images_from_data(
        data={
            'title': '3 batch updates',
            'data': data_to_display
        },
        xlim=(0, 1),
        figsize_mult=(2, 8),
        save_path=Path(
            os.getenv("PWD")
        ) / "images" / "uniform_half_to_1_binomial_mini_batch.png",
    )

    (models[0].prev_accepted_proposals < 0.5).sum()['uniform'] == 0
    (models_full_batch[0].prev_accepted_proposals < 0.5).sum()['uniform'] == 0
    assert (models[0].prev_accepted_proposals < 0.5).sum()['uniform'] == 0
    assert (models_full_batch[0].prev_accepted_proposals < 0.5)\
        .sum()['uniform'] == 0
    assert (models[0].prev_accepted_proposals > 1)\
        .sum()['uniform'] == 0
    assert (models_full_batch[0].prev_accepted_proposals > 1)\
        .sum()['uniform'] == 0


def test_beta_binomial_non_abc_rejection_sampling():
    """
    To see how settings affect the dispersion of the posterior distribution,
    here we vary a bunch of settings. We vary the number of epsilons, whether
    or not to use a constant standard deviation for the perturbation process
    within one abc_smc run, and if not using a constant standard deviation,
    varying how thin the adaptive standard deviation.
    """
    num_particles = 2000
    obs = np.array([
        1, 1, 1,
        1, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 1
    ])

    def distance(x, y):
        """
        Compare the number of ones in one vs. the other.
        """
        return abs(x.sum() - y.sum())

    def simulate(priors, obs):
        """
        Data is binomially distributed.
        """
        return np.random.binomial(n=1, p=priors['beta'], size=len(obs))

    data_to_display = []

    constant_devs = [True, True, True]
    beta_std_divs = [1.0, 2.0, 3.0]
    cols = list(range(len(beta_std_divs)))

    for col, beta_std_div, use_constant_dev in zip(
        cols,
        beta_std_divs,
        constant_devs
    ):

        data_to_display.append([
            {
                'title': f"batch 1: {obs[:3]}"
                + f", std_div: {beta_std_div},"
                + f" constant_dev: {use_constant_dev}",
                'data': []
            },
            {
                'title': f"batch 2: {obs[3:7]},"
                + f" std_div: {beta_std_div},"
                + f" constant_dev: {use_constant_dev}",
                'data': []
            },
            {
                'title': f"batch 3: {obs[7:]},"
                + f" std_div: {beta_std_div},"
                + f" constant_dev: {use_constant_dev}",
                'data': []
            },
            {
                'title': "full batch,"
                + f" std_div: {beta_std_div},"
                + f" constant_dev: {use_constant_dev}",
                'data': []
            }
        ])

        obs_indices = [(0, 3), (3, 7), (7, len(obs))]
        epsilon_sets = [[0], [0], [1, 0], [3, 2, 1, 0]]
        perturbations_config = [False, True, True, True]

        for i, (epsilons, use_perturbation) in enumerate(
            zip(epsilon_sets, perturbations_config)
        ):
            models = Models(
                [
                    Model(
                        name='flat prior',
                        priors=[
                            Beta(
                                alpha=1,
                                beta=1,
                                name="beta",
                            )
                        ],
                        simulate=simulate,
                        prior_model_proba=1,
                        perturb=use_perturbation
                    ),
                ],
                use_constant_dev=use_constant_dev
            )

            # Loop through the rows
            for j, (start_index, end_index) in enumerate(obs_indices):
                obs_batch = obs[start_index:end_index]

                if i == 0:
                    data_to_display[col][j]['data'].append(
                        pd.DataFrame(
                            {
                                'target': np.random.beta(
                                    obs[:end_index].sum() + 1,
                                    len(obs[:end_index])
                                    - obs[:end_index].sum() + 1,
                                    num_particles
                                )
                            }
                        )
                    )
                # Update with 1st batch
                abc_smc(
                    num_particles=num_particles,
                    epsilons=epsilons,
                    models=models,
                    obs=obs_batch,
                    distance=distance,
                )

                data_to_display[col][j]['data'].append(
                    pd.DataFrame(models[0].prev_accepted_proposals)
                    .rename(
                        columns={
                            'beta': f'after batch {j + 1}, perturb: '
                            + f'{use_perturbation}, eps: {epsilons}'
                        }
                    )
                )

                # The posterior distribution becomes the prior
                models.use_distribution_from_samples()

        data_to_display[col][-1]['data'].append(
            pd.DataFrame({
                'reference_posterior': np.random.beta(
                    obs.sum() + 1,
                    len(obs) - obs.sum() + 1,
                    num_particles
                )
            })
        )

        for epsilons in epsilon_sets:
            models = Models(
                [
                    Model(
                        name='flat prior',
                        priors=[
                            Beta(
                                alpha=1,
                                beta=1,
                                name="beta",
                                std_div=beta_std_div,
                            )
                        ],
                        simulate=simulate,
                        prior_model_proba=1,
                    ),
                ],
                use_constant_dev=use_constant_dev
            )

            # Update full batch
            abc_smc(
                num_particles=num_particles,
                epsilons=epsilons,
                models=models,
                obs=obs,
                distance=distance,
            )

            data_to_display[col][-1]['data'].append(
                pd.DataFrame(models[0].prev_accepted_proposals)
                .rename(columns={'beta': f'full batch, eps: {epsilons}'})
            )

    create_images_from_data(
        data={
            'title': '',
            'data': data_to_display
        },
        xlim=(0, 1),
        ylim=(0, 5),
        figsize_mult=(4, 8),
        save_path=Path(
            os.getenv("PWD")
        ) / "images" / "beta_binomial_mini_batch.png",
    )

    assert_similar_enough_distribution(
        models[0].prev_accepted_proposals,
        pd.DataFrame(
            {
                'beta': np.random.beta(
                    obs.sum() + 1, len(obs) - obs.sum() + 1,
                    num_particles
                )
            }
        )
    )
