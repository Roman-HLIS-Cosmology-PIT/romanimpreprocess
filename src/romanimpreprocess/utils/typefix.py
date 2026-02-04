"""Simple fix to output type if we have a different schema."""

import asdf
import numpy as np

def fix(tree):
    """
    Fixes a tree. More fixes can be put here.

    Parameters
    ----------
    tree : AsdfTree
        The ASDF tree to fix in place.

    Returns
    -------
    None

    """

    try:
        tree.validate()
    except asdf._jsonschema.exceptions.ValidationError as ve:
        # this is a common error
        e = str(ve)
        if "'err'" in e and "float16" in e and "err" in tree["roman"]:
            print("Fixing ...", e)
            tree["roman"]["err"] = tree["roman"]["err"].astype(np.float16)
