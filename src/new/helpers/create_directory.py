import os


def create_directory(directory, mode=0o777):
    """
    Create a directory if it does not exist.

    Args:
        directory (str): The directory to create.
        mode (int): The mode to create the directory with.

    Returns:
        bool: True if the directory was created, False if it already exists
        or an error occurred.
    """

    if not os.path.exists(directory):
        oldmask = os.umask(0o000)

        os.makedirs(directory, mode, exist_ok=True)
        os.umask(oldmask)

        return True

    return False
