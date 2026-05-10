"""An artificial case for reference subtraction."""

import numpy as np
from romanimpreprocess.utils.reference_subtraction import ref_subtraction_row


def test_row():
    """Tet for row-based reference subtraction without the use_ref_channel option."""

    im = np.zeros((4096, 4224), dtype=np.float32)
    im[:, :] = np.cos(np.linspace(0, 2000, 4096))[:, None]
    im[:, -128:] *= 2.0
    for x in range(4224):
        im[:, x] += np.sin(0.1 * x) * np.sin(np.linspace(0, 2000, 4096)) ** 3
    im[:, :-128] += 1.0
    im_old = np.copy(im)

    ref_subtraction_row(im, use_ref_channel=False)
    assert np.std(im) < 0.75 * np.std(im_old)
    assert 0.4 < np.std(im[:, :-128]) < 0.5
    assert 0.99 < np.mean(im[:, :-128]) < 1.01
