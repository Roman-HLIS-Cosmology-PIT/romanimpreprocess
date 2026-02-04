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

    max_retries = 4
    for attempt in range(max_retries):
        try:
            tree.validate()
        except asdf._jsonschema.exceptions.ValidationError as ve:
            e = str(ve)

            # this is a common error. move on if we get too many.
            if attempt == max_retries - 1:
                print("Remaining error:", e)
                raise Exception("Validation error reached max tries") from ve

            # Now go through the possible changes we might have to make
            if "'err'" in e and "float16" in e and "err" in tree["roman"]:
                print("Fixing err ...", attempt)
                tree["roman"]["err"] = tree["roman"]["err"].astype(np.float16)
            if "'var_poisson'" in e and "float16" in e and "var_poisson" in tree["roman"]:
                print("Fixing var_poisson ...", attempt)
                tree["roman"]["var_poisson"] = tree["roman"]["var_poisson"].astype(np.float16)
