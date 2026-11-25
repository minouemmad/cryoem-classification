"""Figure per trajectory: PC1 histogram with the 20 bin centres above the 20 rendered maps.
Usage: fig.py DATASET_DIR  -> DATASET_DIR/traj20/figs/traj20_{lat,vol}.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image

T = Path(sys.argv[1]) / "traj20"; (T / "figs").mkdir(exist_ok=True)
INK, MUTED = "#2E3338", "#62666A"
plt.rcParams.update({"font.size": 10, "text.color": INK, "xtick.color": MUTED})
ST = json.load(open(T / "stats.json"))
CROP = (175, 55, 525, 625)
for sp, col, xl in (("lat", "#4A5D7E", "latent PC1 (z)"), ("vol", "#2A9D8F", "volume PC1 (decoded maps)")):
    x = np.load(T / f"scores_{sp}.npy").astype(float); mu, sd = x.mean(), x.std(); x = (x - mu) / sd
    cen = (np.array(ST["traj"][sp]["centres"]) - mu) / sd
    fig = plt.figure(figsize=(12.4, 3.0), dpi=200)
    axh = fig.add_axes([0.02, 0.1, 0.96, 0.12])
    bins = np.linspace(np.percentile(x, 0.1), np.percentile(x, 99.9), 140)
    axh.hist(x, bins=bins, color="#C9CED6", lw=0)
    for c in cen:
        axh.axvline(c, color=col, lw=1.0)
    axh.set_yticks([]); [axh.spines[k].set_visible(False) for k in ("top", "right", "left")]
    axh.set_xlabel(f"particle position along {xl}  (sd)", fontsize=9.5); axh.set_xlim(bins[0], bins[-1])
    for i in range(20):
        a = fig.add_axes([0.02 + i * 0.048, 0.33, 0.048, 0.66]); a.axis("off")
        im = Image.open(T / f"renders/{sp}_{i + 1:03d}.png").crop(CROP); a.imshow(im)
        fig.add_artist(ConnectionPatch(xyA=(im.width / 2, im.height), coordsA=a.transData, xyB=(cen[i], axh.get_ylim()[1]),
                                       coordsB=axh.transData, color=col, lw=0.7, alpha=0.7))
        a.text(0.5, 1.0, f"{i + 1}", transform=a.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
    fig.savefig(T / f"figs/traj20_{sp}.png", bbox_inches="tight", transparent=True); plt.close(fig)
print("ok", T / "figs")
