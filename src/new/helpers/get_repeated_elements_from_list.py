from collections import Counter
from typing import List


def get_repeated_elements_from_list(lst: List, sort: bool = True) -> List:
    """
        This function takes a list and returns a list of elements
        that occur more than once.

        Parameters:
            lst (List): A list of elements.

            sort (bool, optional): Whether to sort the repeated elements.
                Defaults to True.

        Returns:
            List: A list of repeated elements in the list.
    """

    counter = Counter(lst)

    repeated_elements = [{element: count}
                         for element, count in counter.items() if count > 1]

    return repeated_elements if not sort else sorted(
        repeated_elements, key=lambda x: list(x.values())[0], reverse=True)
