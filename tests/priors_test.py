"""
Test prior objects
"""
from ..unlikely.priors import UniformFromSamples


def test_uniform_from_samples():
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    unif_from_samples = UniformFromSamples(
        samples,
        {
            'lower_bound': 0,
            'upper_bound': 0.5
        }
    )

    assert unif_from_samples.pdf(-0.0001) == 0
    assert unif_from_samples.pdf(0) > 0
    assert unif_from_samples.pdf(0.5) > 0
    assert unif_from_samples.pdf(0.50001) == 0
