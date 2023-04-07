from typing import List

from .get_flattened_list import get_flattened_list


def get_diff_elements_from_list_of_lists(lst: List[List],
                                         sort: bool = True) -> List:
    """
    Returns a list of elements that appear in some, but not all, of the
    sublists in the given list.

    Parameters:
        lst (List[List]): A list of sublists.

    Returns:
        List: A list of elements that appear in some, but not all, of the
        sublists in the given list.
    """

    flattened_lst = get_flattened_list(lst)

    elem_counts = {}
    for elem in flattened_lst:
        if elem not in elem_counts:
            elem_counts[elem] = 0
        elem_counts[elem] += 1

    diff_elems = [
        elem for elem in elem_counts if elem_counts[elem] != len(lst)]

    return diff_elems if not sort else sorted(diff_elems)
