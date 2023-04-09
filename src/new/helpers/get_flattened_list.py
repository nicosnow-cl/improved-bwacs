from typing import List


def get_flattened_list(lst: List[List], elem_type=None):
    """
    Returns a flattened version of the given list of lists.

    Parameters:
        lst (List[List]): A list of sublists.

    Returns:
        List: A flattened version of the given list of lists.
    """

    return [elem if elem_type is None else elem_type(elem) for sublist in lst
            for elem in sublist]
