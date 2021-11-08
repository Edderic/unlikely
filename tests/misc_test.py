import numpy as np
import pytest
from scipy.stats import beta
from ..unlikely.misc import find_closest, hpdi


def test_hpdi():
    size = 10_000
    # Skewed to the right
    array = np.random.beta(2, 1, size=size)
    credible_interval_width = 0.95
    lower, upper = hpdi(proba=credible_interval_width, array=array)
    beta_dist = beta(2, 1)
    probability = beta_dist.cdf(upper) - beta_dist.cdf(lower)
    assert credible_interval_width == pytest.approx(probability, abs=0.01)
    assert 1 == pytest.approx(upper, abs=0.01)

    # Skewed to the left
    array = np.random.beta(1, 2, size=size)
    credible_interval_width = 0.95
    lower, upper = hpdi(proba=credible_interval_width, array=array)
    beta_dist = beta(1, 2)
    probability = beta_dist.cdf(upper) - beta_dist.cdf(lower)
    assert credible_interval_width == pytest.approx(probability, abs=0.01)
    assert 0 == pytest.approx(lower, abs=0.01)


def test_find_closest():
    assert find_closest(
        to_find=1,
        values=[0, 1, 2, 3],
        left_index=0,
        right_index=3
    ) == 1

    # 4.4 is closer to 4 than 5
    assert find_closest(
        to_find=4.4,
        values=[0, 1, 2, 3, 4, 5, 6],
        left_index=0,
        right_index=6
    ) == 4

    # 4.5 is equally close to 4 and 5. Let's pick the bigger value.
    assert find_closest(
        to_find=4.5,
        values=[0, 1, 2, 3, 4, 5, 6],
        left_index=0,
        right_index=6
    ) == 5

    # 4.6 is closer to 5 than 4, so we pick 5.
    assert find_closest(
        to_find=4.6,
        values=[0, 1, 2, 3, 4, 5, 6],
        left_index=0,
        right_index=6
    ) == 5

    # Test out of bounds left
    assert find_closest(
        to_find=-1,
        values=[0, 1, 2, 3, 4, 5, 6],
        left_index=0,
        right_index=6
    ) == 0

    # Test out of bounds right
    assert find_closest(
        to_find=100,
        values=[0, 1, 2, 3, 4, 5, 6],
        left_index=0,
        right_index=6
    ) == 6
