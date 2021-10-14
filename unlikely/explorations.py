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

def plot_bayesian_updating_beta_example(save_path, client=None):
    """
    This function compares using ABC-SMC for iterative, one-by-one Bayesian
    updating vs. doing Bayesian updating with the full data set (i.e. all at
    once). I found out that using a Normal distribution as the perturbation
    kernel, along with a Normal distribution for the weighting scheme could
    cause overdispersion of the posterior distribution in iterative one-by-one
    updating, compared to the reference posterior distribution.

    Parameters:
        save_path: Path-like object
        client: dask.distributed.Client. Defaults to None
            client can be one that uses multiple workers so that we can
            parallelize this work.

    I used the globe-tossing example from Richard McElreath's Statistical
    Rethinking, 2nd Ed. In this example, we are trying to figure out what the
    percentage of water is on Earth, and we collected 9 binary data points
    where each data point is either "water" or "land". We could model this
    with Beta-Binomial model. We can think of a probability of 0 being the
    Earth is completely land, while 1 is completely water. The beta
    distribution is used for the prior, since it constrains the possibilities
    between 0 and 1. On the other hand, the binomial distribution is used for
    the likelihood of observing the binary observations, where an observation 1
    denotes water, while an observation 0 denotes observing land. The binomial
    distribution is conjugate to the beta prior (i.e. the resulting posterior
    distribution is another beta distribution). Conjugate pairings have closed
    form solutions that let us efficiently compute the posterior
    distribution, which in turn allows us to check our work. In this example,
    the posterior distribution can efficiently be computed via Beta(alpha=x_0 +
    x_1, beta=y_0 + y_1), where x_0, and x_1 represent our prior belief in the
    number of successses x_0 and the number of failures y_0. x_1 and y_1
    represent the observed data points, where x_1 is the number of successes
    and y_1 is the number of failures in the data.

    In this example, we tossed the globe 9 times and our "finger" lands on 6
    water and 3 land. Here's how we can exploiting the conjugate relationship
    above'Starting with a Uniform prior Beta(alpha=1, beta=1) (i.e.  a uniform
    prior), we could get a posterior distribution via Beta(alpha=1+6, beta=1+3)
    = Beta(alpha=7, beta=4). In other words, the posterior distribution is
    summing the successes (from data and prior) and then summing the number of
    failures (also from data and prior). Beta(alpha=7, beta=4) is then our
    reference distribution.

    The first column represents one-by-one updating with exact rejection
    sampling. Doing this process gives us a posterior distribution that has
    good performance in reproducing the reference distribution. However, if we
    use perturbation like in the original ABC-SMC scheme of Toni et al. (2009),
    then the variance of the posterior distribution becomes inflated (see 2nd
    column). The third column shows updating with all the data at once. We have
    two rows with differing lengths for lists of epsilons. Both of them reach
    similar posterior distributions, and those are pretty similar to the
    reference posterior distribution.

    TODO: try one-by-one updating with perturbation, but have a long list of
    epsilons 0,0,0,0.
    """

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

    models = [
        models_no_perturb,
        models_perturb,
    ]

    epsilons = [
        [0],
        [0],
    ]

    for ms, eps in zip(models, epsilons):
        abc_smc(
            num_particles=num_particles,
            epsilons=eps,
            models=ms,
            obs=np.array([obs[0]]),
            distance=distance,
            client=client
        )

    model_mappings = {
        'perturb': {
            'models': models_perturb,
            'epsilons': [0],
        },
        'no_perturb': {
            'models': models_no_perturb,
            'epsilons': [0]
        }
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
        for ms_type, model_meta in model_mappings.items():
            ms = model_meta['models']
            epsilons = model_meta['epsilons']

            ms.use_distribution_from_samples()

            abc_smc(
                num_particles=num_particles,
                epsilons=epsilons,
                models=ms,
                obs=np.array([obs[i]]),
                distance=distance,
                client=client
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
