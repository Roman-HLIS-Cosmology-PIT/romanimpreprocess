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

    # Add in dummy chi^2 and dumo for now (roman-hlis-l2-driver doesn't use this)
    new_fields = ["chisq", "dumo"]
    for fld in new_fields:
        if fld not in tree["roman"]:
            tree["roman"][fld] = np.zeros(np.shape(tree["roman"]["data"]), dtype=np.float16)
            if "dummyfields" not in tree["roman"]["meta"]:
                tree["roman"]["meta"]["dummyfields"] = []
            tree["roman"]["meta"]["dummyfields"].append(f"roman.{fld}")
    if "dummyfields" in tree["roman"]["meta"]:
        print("added dummy fields:", tree["roman"]["meta"]["dummyfields"])

    # Fixing error that occurs with wfi parallel flag when using new roman datamodels with new utilities
    if "wfi_parallel" not in tree["roman"]["meta"]["observation"]:
        tree["roman"]["meta"]["observation"]["wfi_parallel"] = False

    # Which fields to check in "roman"
    changetypes = {"err": "float16", "var_poisson": "float16", "var_rnoise": "float16", "var_flat": "float16"}

    max_retries = len(changetypes.keys()) + 2
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
            for fld in changetypes:
                if f"'{fld}'" in e and changetypes[fld] in e and fld in tree["roman"]:
                    print("Fixing", fld, "...", attempt)
                    tree["roman"][fld] = tree["roman"][fld].astype(np.dtype(changetypes[fld]))
