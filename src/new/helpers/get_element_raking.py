from random import random


def get_element_ranking(elem, base_rank, iterables, pseudo_random=False):
    """
    Get the ranking of an element in a list of iterables.

    Args:
        elem (any): Element to get the ranking.
        base_rank (int): Base ranking.
        iterables (list): List of iterables.
        pseudo_random (bool): If True, the ranking will be pseudo-random.

    Returns:
        int: Ranking of the element.
    """

    ranking = base_rank

    for iterable in iterables:
        if elem in iterable:
            if pseudo_random:
                ranking += ranking * random()
            else:
                ranking += ranking

    return ranking
