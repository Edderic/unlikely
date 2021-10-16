import pytest # pylint:disable=F401
import pdb

DEBUG_FAILURES = True

def assert_similar_enough_distribution(
    actual,
    expected,
    quantiles=None,
    tolerance=None
):
    """
    Test that the distributions are "close" enough.
    Compares quantiles of accepted proposals and expected

    Parameters:
        actual: a distribution
            responds to ".quantile" method.

        expected: a distribution
            responds to ".quantile" method.

        quantiles: list of floats
            A list of floats between 0 and 1 (inclusive).

        tolerance: float
            Positive value or 0 denoting what is acceptable for the absolute
            distance between the quantiles.
    """

    if quantiles is None:
        quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]

    if tolerance is None:
        tolerance = 0.1

    actual_quantiles = actual.quantile(quantiles)
    expected_quantiles = expected.quantile(quantiles)

    absolute_diff = abs(
        actual_quantiles - expected_quantiles
    )

    try:
        assert (absolute_diff < tolerance)\
            .sum().loc[expected.columns[0]] == actual_quantiles.shape[0]
    except AssertionError as e:
        if DEBUG_FAILURES:
            pdb.set_trace()

        raise e
