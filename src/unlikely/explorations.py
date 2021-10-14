from pathlib import Path

import numpy as np
import pandas as pd

from .engine import abc_smc
from .misc import save_images_from_data
from .models import Models, Model
from .priors import Beta

def distance(x,y):
    return abs(x.sum() - y.sum())

def simulate(priors, size=1):
    return np.random.binomial(n=1, p=priors['beta'], size=size)

def plot_bayesian_updating_beta_example(save_path):
    client = None

    obs = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1])
    obs_str = []
    for i in obs:
        if i:
            obs_str.append('W')
        else:
            obs_str.append("L")

    num_particles = 2000

    models_no_perturb = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                epsilon_len=len([0]),
                simulate=simulate,
                prior_model_proba=1,
                perturb=False
            ),
        ],
        perturbation_param=0.9
    )

    models_perturb = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                epsilon_len=len([0]),
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
        perturbation_param=0.9
    )

    for ms in [models_no_perturb, models_perturb]:
        abc_smc(
            num_particles=num_particles,
            epsilons=[0],
            models=ms,
            obs=np.array([obs[0]]),
            distance=distance,
            client=client
        )


    accepted_proposals_description = {
        'perturb': [models_perturb[0].prev_accepted_proposals.describe()],
        'no_perturb': [models_no_perturb[0].prev_accepted_proposals.describe()]
    }

    # Problem: Original paper assumes that the method only runs once. Maybe we
    # should have the model object have an idea of how many epochs it has
    # processed? (stateful) Would it store the whole set of variables from the
    # beginning of time?
    # Good for debugging, but that would risk having memory issues. Maybe we
    # could offload the data at a later point?
    # Prior

    # Online training
    # We have a new observation
    # Use the last iteration as new prior distributions.
    for i in range(1, len(obs)):
        for ms, ms_type in zip(
            [models_no_perturb, models_perturb],
            ["no_perturb", "perturb"]
        ):
            ms.use_distribution_from_samples()

            abc_smc(
                num_particles=num_particles,
                epsilons=[0],
                models=ms,
                obs=np.array([obs[i]]),
                distance=distance,
                client=client
            )

            accepted_proposals_description[ms_type].append(
                ms[0].prev_accepted_proposals.describe()
            )

    # All at once
    models_all_at_once_3_2_1_0 = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                epsilon_len=len([3,2,1,0]),
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
        perturbation_param=0.9
    )

    abc_smc(
        num_particles,
        epsilons=[3,2,1,0],
        models=models_all_at_once_3_2_1_0,
        obs=obs,
        distance=distance,
        client=client
    )

    # All at once, 5-0 eps
    models_all_at_once_5_0_eps = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                epsilon_len=len([5,4,3,2,1,0]),
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
        perturbation_param=0.9
    )

    abc_smc(
        num_particles,
        epsilons=[5,4,3,2,1,0],
        models=models_all_at_once_5_0_eps,
        obs=obs,
        distance=distance,
        client=client
    )

    # All at once, 0 eps
    models_all_at_once_0_eps = Models(
        [
            Model(
                name='flat prior',
                priors=[
                    Beta(alpha=1, beta=1, name="beta"),
                ],
                epsilon_len=len([0]),
                simulate=simulate,
                prior_model_proba=1
            ),
        ],
        perturbation_param=0.9
    )

    abc_smc(
        num_particles,
        epsilons=[0],
        models=models_all_at_once_0_eps,
        obs=obs,
        distance=distance,
        client=client
    )

    expected = pd.DataFrame(
        {'expected': np.random.beta(7,4,num_particles)}
    )

    updates_no_perturb = [
        {
            'title': f"{''.join(obs_str[:i+1])}{''.join(obs_str[i+1:]).lower()}",
            'data': [ pd.DataFrame(d).rename(columns={'beta': 'not perturbed, 1-by-1'}) ]
        } for i, d in enumerate(
            models_no_perturb[0].accepted_proposals
        )

    ]

    updates_no_perturb[-1]['data'].append(expected)

    updates_perturb = [
        {
            'title': f"{''.join(obs_str[:i+1])}{''.join(obs_str[i+1:]).lower()}",
            'data': [ pd.DataFrame(d).rename(columns={'beta': 'perturbed, 1-by-1'}) ]
        } for i, d in enumerate(
            models_perturb[0].accepted_proposals
        )

    ]

    updates_perturb[-1]['data'].append(expected)
    graph_data = {
        'title': "One-by-one vs. Full Updating, n=2000",
        'data': [
            updates_no_perturb,
            updates_perturb,
            [
                {
                    "title": "Perturbed, full. eps: 5,4,3,2,1,0",
                    "data": [
                        models_all_at_once_5_0_eps[0].prev_accepted_proposals,
                        expected
                    ],
                },
                {
                    'title': "Perturbed, full. eps: 3,2,1,0",
                    'data': [
                        models_all_at_once_3_2_1_0[0].prev_accepted_proposals,
                        expected
                    ],
                },
                {
                    'title': "Perturbed, full. eps: 0",
                    'data': [
                        models_all_at_once_0_eps[0].prev_accepted_proposals,
                        expected
                    ]
                }
            ]
        ]
    }

    save_images_from_data(
        save_path=save_path,
        data=graph_data,
        xlim=(0,1),
    )


if __name__ == '__main__':
    save_path = Path("/Users/eugaddan/Desktop/beta.png")
    plot_bayesian_updating_beta_example(save_path=save_path)
