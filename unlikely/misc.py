import logging
from pathlib import Path

import matplotlib.pyplot as plt

def create_images_from_data(
    data,
    xlim,
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
    if bins is None:
        bins = False

    if figsize_mult is None:
        figsize_mult = (2,5)

    num_rows = 0
    for col in data['data']:
        if len(col) > num_rows:
            num_rows = len(col)

    num_cols = len(data['data'])

    x, y = figsize_mult
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_rows * x, num_cols * y)
    )

    fig.suptitle(data['title'])

    for col_index, col in enumerate(data['data']):
        for row_index, row in enumerate(col):

            if num_cols == 1 and num_rows > 1:
                ax = axs[row_index]
            elif num_cols == 1 and num_rows == 1:
                ax = axs
            else:
                ax = axs[row_index, col_index]

            ax.set_xlim(xlim)
            ax.set_title(row['title'])

            for df in row['data']:
                if bins:
                    df.plot.hist(
                        ax=ax,
                        alpha=alpha,
                        bins=bins
                    )
                else:
                    df.plot.kde(
                        ax=ax,
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

def distance(x,y):
    """
    Get absolute value between x and y.

    Parameters:
        x: numeric
        y: numeric

    Returns: numericc
    """
    return abs(x-y)

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
    if values[left_index] == to_find:
        return left_index
    if values[right_index] == to_find:
        return right_index

    if right_index - left_index <= 1:
        if distance(to_find, values[left_index]) >= distance(to_find, values[right_index]):
            return right_index

        return left_index

    if distance(to_find, values[left_index]) >= distance(to_find, values[right_index]):
        return find_closest(
            to_find,
            values,
            left_index=(right_index - left_index) // 2 + left_index,
            right_index=right_index
        )

    if distance(to_find, values[left_index]) < distance(to_find, values[right_index]):
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
