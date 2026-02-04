"""
Simple utility to save data type.

This was introduced to avoid collisions with different versions of schemas.

"""

import asdf  # will want later  # noqa: F401
import numpy as np

class TypeRef:
    """
    Saves data type information.

    Parameters
    ----------
    tree : dict or dict-like
        The tree (intended to be an ASDF tree).

    """

    # The items to track (use . as separator)
    items = ["roman.err", "wont_need_this", "wont_need_this.wont_need_this_either"]

    def __init__(self, tree):
        self.types = {}
        for item in self.items:
            r = tree
            hierarchy = item.split(".")
            n = len(hierarchy)
            for j in range(n):
                if hierarchy[j] in r:
                    r = r[hierarchy[j]]
                else:
                    break
                if j == n - 1 and isinstance(r, np.ndarray):
                    self.types[item] = r.dtype

    def convert(self, tree):
        """
        Converts items in the tree to the data type passed to the constructor.

        Parameters
        ----------
        tree : dict or dict-like
            The tree (intended to be an ASDF tree).

        Returns
        -------
        None

        """

        for item in self.items:
            r = tree
            hierarchy = item.split(".")
            n = len(hierarchy)
            for j in range(n):
                if hierarchy[j] in r:
                    rprev = r
                    r = r[hierarchy[j]]
                else:
                    break
                if j == n - 1 and isinstance(r, np.ndarray):
                    rprev[hierarchy[j]] = r.astype(self.types[item])
