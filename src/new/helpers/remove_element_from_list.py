from typing import Any, List


def remove_element_from_list(lst: List, elem: Any) -> List:
    """
        Returns a new list with all occurrences of the given element removed.

        Parameters:
            lst (List): The list to remove elements from.

            elem (Any): The element to remove from the list.

        Returns:
            List: A new list with all occurrences of the given element removed.
    """

    return [value for value in lst if value != elem]
