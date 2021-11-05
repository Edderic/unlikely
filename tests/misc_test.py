import numpy as np
import pytest
from ..unlikely.misc import find_closest, hpdi


def test_hpdi():
    # Skewed to the right, but already sorted
    array = np.array(np.arange(1000))
    lower, upper = hpdi(proba=0.95, array=array)

    assert lower == 49
    assert upper == 999

    # Skewed to the right, but sorted in reverse
    array = np.arange(1000)[::-1]
    lower, upper = hpdi(proba=0.95, array=array)
    assert lower == 49
    assert upper == 999


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
