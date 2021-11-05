"""
Miscellaneous functions
"""
import logging

import matplotlib.pyplot as plt


def create_images_from_data(
    data,
    xlim,
    ylim=None,
    save_path=None,
    alpha=0.5,
    bins=None,
    figsize_mult=None
):
    """
    Parameters:
        data: dict

        xlim: tuple
            X limits

        ylim: tuple
            Y limits

        save_path: Path-like object.
            If None, this function will return a matplotlib figure and axes.

        alpha: float
            Transparency of items.

        bins: Defaults to False

        figsize_mult: tuple
            2-item tuple, where each one is an integer. We use
            these as part of figsize.

    Returns: tuple
        fig, ax: matplotlib objects

    Example:
    >>> import numpy as np
    >>> from unlikely.misc import create_images_from_data
    >>> series_1 = pd.DataFrame({'leg1': np.random.uniform(1000)})
    >>> series_2 = pd.DataFrame({'leg2': np.random.uniform(1000)})
    >>> series_3 = pd.DataFrame({'leg3': np.random.uniform(1000)})
    >>> series_4 = pd.DataFrame({'leg4': np.random.uniform(1000)})
    >>> series_5 = pd.DataFrame({'leg5': np.random.uniform(1000)})
    >>> series_6 = pd.DataFrame({'leg6': np.random.uniform(1000)})
    >>> series_7 = pd.DataFrame({'leg7': np.random.uniform(1000)})
    >>> series_8 = pd.DataFrame({'leg8': np.random.uniform(1000)})
    >>> create_images_from_data(
    >>>     data={
    >>>         'title': "Figure super title",
    >>>         'data': [
    >>>             [
    >>>                 {
    >>>                     'title': 'Row 1, Col 1'
    >>>                     'data': [series_1, series_2]
    >>>                 },
    >>>                 {
    >>>                     'title': 'Row 2, Col 1'
    >>>                     'data': [series_3, series_4]
    >>>                 }
    >>>             ],
    >>>             [
    >>>                 {
    >>>                     'title': 'Row 1, Col 2'
    >>>                     'data': [series_5, series_6]
    >>>                 },
    >>>                 {
    >>>                     'title': 'Row 2, Col 2'
    >>>                     'data': [series_7, series_8]
    >>>                 }
    >>>             ]
    >>>         ]
    >>>     }
    >>> )

    """
    # pylint:disable=too-many-locals,too-many-arguments,too-many-branches
    if bins is None:
        bins = False

    if figsize_mult is None:
        figsize_mult = (2, 5)

    num_rows = 0
    for col in data['data']:
        if len(col) > num_rows:
            num_rows = len(col)

    num_cols = len(data['data'])

    row_mult, col_mult = figsize_mult
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_rows * row_mult, num_cols * col_mult)
    )

    fig.suptitle(data['title'])

    for col_index, col in enumerate(data['data']):
        for row_index, row in enumerate(col):

            if num_cols == 1 and num_rows > 1:
                axis = axs[row_index]
            elif num_cols == 1 and num_rows == 1:
                axis = axs
            else:
                axis = axs[row_index, col_index]

            axis.set_xlim(xlim)
            axis.set_title(row['title'])

            if ylim is not None:
                axis.set_ylim(ylim)

            for dataframe in row['data']:
                if bins:
                    dataframe.plot.hist(
                        ax=axis,
                        alpha=alpha,
                        bins=bins
                    )
                else:
                    dataframe.plot.kde(
                        ax=axis,
                        alpha=alpha
                    )

    fig.set_tight_layout(True)

    if save_path:
        fig.savefig(
            save_path,
            format='png',
            bbox_inches='tight'
        )

    return fig, axs


def distance(arg1, arg2):
    """
    Get absolute value between arg1 and arg2.

    Parameters:
        arg1: numeric
        arg2: numeric

    Returns: numericc
    """
    return abs(arg1-arg2)


def find_index_of_closest(to_find, values):
    """
    Find index of closest value from a list of values

    Parameters:
        to_find: float
            The value to find. Doesn't necessarily have to be in there.

        values: list
            sorted
    """
    return find_closest(to_find, values, 0, len(values) - 1)


def find_closest(to_find, values, left_index, right_index):
    """
    Private method for recursion.

    Parameters:
        to_find: float
            The value to find. Doesn't necessarily have to be in there.

        values: list
            sorted

        left_index: integer
        right_index: integer
    """
    if values[left_index] == to_find:
        return left_index
    if values[right_index] == to_find:
        return right_index

    if right_index - left_index <= 1:
        if distance(
            to_find, values[left_index]
        ) >= distance(to_find, values[right_index]):
            return right_index

        return left_index

    if distance(
        to_find, values[left_index]
    ) >= distance(to_find, values[right_index]):
        return find_closest(
            to_find,
            values,
            left_index=(right_index - left_index) // 2 + left_index,
            right_index=right_index
        )

    return find_closest(
        to_find,
        values,
        left_index=left_index,
        right_index=(right_index - left_index) // 2 + left_index
    )


def setup_logging(filename=None, filemode='a', level=logging.INFO):
    """
    Sets up logging with time, levelname, and message.

    Returns: logging
    """
    args = {
        'format': '%(asctime)s %(levelname)-8s %(message)s',
        'level': level,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

    if filename:
        args['filename'] = filename
        args['filemode'] = filemode

    logging.basicConfig(**args)

    return logging
