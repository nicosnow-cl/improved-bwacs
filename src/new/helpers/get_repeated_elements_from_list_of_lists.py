from typing import List

from .get_repeated_elements_from_list import get_repeated_elements_from_list


def get_repeated_elements_from_list_of_lists(lst: List[List],
                                             sort: bool = True) -> List:
    """
        This function takes a list of sublists and returns a list of elements
        that occur more than once in the sublists.

        Parameters:
            lst (List[List]): A list of sublists.

            sort (bool, optional): Whether to sort the repeated elements.
                Defaults to True.

        Returns:
            List: A list of repeated elements in the sublists.
    """

    flat_list = [item for sublist in lst for item in sublist]

    return get_repeated_elements_from_list(flat_list, sort)
