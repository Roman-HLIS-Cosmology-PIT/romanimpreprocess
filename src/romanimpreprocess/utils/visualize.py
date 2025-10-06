"""Visualization script."""

import sys

import asdf

# plotting
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.switch_backend("agg")

import matplotlib.colors as colors  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("This is a simple visualization script.")
        print("Calling format: python visualize.py infile.asdf xmin,xmax,ymin,ymax outfile.pdf [percentile_cut]")
        exit()
    
    bounds = sys.argv[2].split(",")
    xmin = int(bounds[0])
    xmax = int(bounds[1])
    ymin = int(bounds[2])
    ymax = int(bounds[3])
    dx = xmax - xmin + 1
    dy = ymax - ymin + 1
    
    with asdf.open(sys.argv[1]) as f:
        data = f["roman"]["data"][:, ymin : ymax + 1, xmin : xmax + 1].astype(np.float32)
    ng = np.shape(data)[0]
    
    matplotlib.rcParams.update({"font.size": 8})
    F = plt.figure(figsize=(3.5 * ng, 6))
    
    percentile_cut = 2.0
    if len(sys.argv) > 4:
        percentile_cut = float(sys.argv[4])
    
    # the main images
    vmin = np.percentile(data, percentile_cut)
    vmax = np.percentile(data, 100 - percentile_cut)
    for j in range(ng):
        S = F.add_subplot(2, ng, 1 + j)
        S.set_title(rf"Group {j:d}")
        S.set_xlabel(f"x-{xmin:d}")
        S.set_ylabel(f"y-{ymin:d}")
        im = S.imshow(
            data[j, :, :], cmap="magma", aspect=1.0, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax
        )
        F.colorbar(im, orientation="vertical", fraction=0.046, pad=0.04)
    
    # the differences
    diff = data - data[1, :, :][None, :, :]
    S = F.add_subplot(2, ng, ng + 1)
    S.set_title(r"Grp0-Grp1")
    S.set_xlabel(f"x-{xmin:d}")
    S.set_ylabel(f"y-{ymin:d}")
    vmax = np.percentile(diff[0, :, :], 100 - percentile_cut)
    vmin = np.percentile(diff[0, :, :], percentile_cut)
    im = S.imshow(
        diff[0, :, :], cmap="magma", aspect=1.0, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax
    )
    F.colorbar(im, orientation="vertical", fraction=0.046, pad=0.04)
    
    vmax = np.percentile(diff[-1, :, :], 100 - percentile_cut)
    vmin = -0.05 * vmax
    for j in range(2, ng):
        S = F.add_subplot(2, ng, ng + 1 + j)
        S.set_title(rf"Grp{j:d}-Grp1")
        S.set_xlabel(f"x-{xmin:d}")
        S.set_ylabel(f"y-{ymin:d}")
        im = S.imshow(
            diff[j, :, :],
            cmap="magma",
            aspect=1.0,
            interpolation="nearest",
            origin="lower",
            norm=colors.PowerNorm(gamma=2.0 / 3.0, vmin=vmin, vmax=vmax),
        )
        F.colorbar(im, orientation="vertical", fraction=0.046, pad=0.04)
    
    F.set_tight_layout(True)
    F.savefig(sys.argv[3])
    plt.close(F)
