"""Shared helpers: latents, decoded volumes, class sets, 1-D splits, volume features, JSON / pickle I/O.
Only load trusted cryoDRGN pickle files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans

from cryodrgn.mrcfile import parse_mrc



def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while reading: {path}")
    return dict(path=str(path.resolve()), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())


def empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty (no files overwritten): {path}")


def load_latents(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        # train_vae writes posterior means first, followed by log variances.
        z = np.asarray(pickle.load(stream), dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if z.ndim != 2 or min(z.shape) == 0 or not np.isfinite(z).all():
        raise ValueError("Latents must be a nonempty, finite N x zdim array")
    return z


def sample_latents(z: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 3 or count > len(z):
        raise ValueError(f"--samples must be between 3 and {len(z)}")
    if len(np.unique(z, axis=0)) < count:
        raise ValueError("Fewer distinct latent vectors than requested samples")
    if count == len(z):
        return np.arange(count), np.ones(count, dtype=int)
    clustering = MiniBatchKMeans(
        n_clusters=count, batch_size=max(4096, count * 3), n_init=3,
        random_state=seed, reassignment_ratio=0,
    ).fit(z)
    labels = clustering.predict(z)
    counts = np.bincount(labels, minlength=count)
    if np.any(counts == 0):
        raise ValueError("Empty sampling clusters; reduce --samples or change --seed")
    indices = []
    for cluster in range(count):
        rows = np.flatnonzero(labels == cluster)
        distances = np.sum((z[rows] - clustering.cluster_centers_[cluster]) ** 2, axis=1)
        indices.append(int(rows[np.argmin(distances)]))
    return np.asarray(indices), counts


def read_volume(path: Path, box: int, apix: float) -> np.ndarray:
    volume, header = parse_mrc(str(path))
    if volume.shape != (box, box, box) or not np.isrealobj(volume):
        raise ValueError(f"{path}: expected a real {box}^3 volume, got {volume.shape}")
    fields = header.fields
    spacing = [fields[f"{axis}len"] / fields[f"m{axis}"] for axis in "xyz"]
    if not np.allclose(spacing, apix, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{path}: voxel size {spacing} does not match {apix}")
    if [fields[key] for key in ("mapc", "mapr", "maps")] != [1, 2, 3]:
        raise ValueError(f"{path}: nonstandard map axes; resample into the decoder frame")
    if any(fields[key] != 0 for key in ("xorg", "yorg", "zorg", "nxstart", "nystart", "nzstart")):
        raise ValueError(f"{path}: nonzero origin/start; use uncropped decoder-frame maps")
    if not np.isfinite(volume).all():
        raise ValueError(f"{path}: nonfinite density")
    return np.asarray(volume, dtype=np.float32)


def fresh(path: Path) -> Path:
    path = path.resolve()
    empty_output(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_class_set(spec: list[str], uid: np.ndarray) -> tuple[str, list[str], np.ndarray]:
    """spec = [SETNAME, LABEL=path.cs, ...] -> hard label per particle (-1 = none)."""
    order = np.argsort(uid)
    lab = np.full(len(uid), -1)
    names = []
    for j, item in enumerate(spec[1:]):
        name, path = item.split("=", 1)
        u = np.load(path)["uid"]
        pos = np.searchsorted(uid[order], u)
        if np.any(pos >= len(uid)) or not np.array_equal(uid[order][pos], u):
            raise ValueError(f"{path}: UIDs not all in the particle set")
        if np.any(lab[order[pos]] >= 0):
            raise ValueError(f"{path}: particle in two classes")
        lab[order[pos]] = j
        names.append(name)
    return spec[0], names, lab


def weighted_pc1(F: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population-weighted mean and top principal direction of centre features (rows)."""
    w = w / w.sum()
    mu = w @ F
    X = (F - mu) * np.sqrt(w)[:, None]
    ev, U = np.linalg.eigh(X @ X.T)  # Gram trick: rows (centres) << voxels
    v = X.T @ U[:, -1]
    return mu, v / np.linalg.norm(v), ev[::-1][:3] / ev.sum()


def kmeans1d(x: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    """Thresholds of a 1-D k-means (quantile initialisation); k-1 increasing cut points."""
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        thr = 0.5 * (c[1:] + c[:-1])
        lab = np.searchsorted(thr, x)
        new = np.array([x[lab == j].mean() for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return 0.5 * (c[1:] + c[:-1])


def volume_features(paths, mask, box, apix, smooth_fwhm, metric, max_memory_gb):
    if not np.isfinite(smooth_fwhm) or smooth_fwhm < 0 or max_memory_gb <= 0:
        raise ValueError("Smoothing must be >=0 and memory budget >0")
    support = mask > 0
    weights = mask[support].astype(np.float64)
    if len(weights) < 2 or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("Use a nonempty soft mask with values in [0,1]")
    estimated = 4 * 4 * len(paths) * len(weights) + 8 * len(paths) ** 2
    if estimated > max_memory_gb * 2**30:
        raise ValueError(f"Estimated working memory {estimated / 2**30:.2f} GiB exceeds budget")
    features = np.empty((len(paths), len(weights)), dtype=np.float32)
    sigma = smooth_fwhm / (np.sqrt(8 * np.log(2)) * apix)
    for i, path in enumerate(paths):
        volume = read_volume(path, box, apix)
        if sigma:
            volume = gaussian_filter(volume, sigma=sigma, mode="constant")
        values = volume[support].astype(np.float64)
        if metric == "correlation":
            values -= np.average(values, weights=weights)
        values *= np.sqrt(weights / weights.sum())
        if metric == "correlation":
            norm = np.linalg.norm(values)
            if norm <= np.finfo(float).eps:
                raise ValueError(f"Constant map within mask: {path}")
            values /= norm
        features[i] = values
    return features


def cut_points(s, K, method, rng):
    """K-1 increasing cut points on the 1-D scores s, or None when the method finds no K-way cut."""
    if method == "kmeans":
        return kmeans1d(s, K)
    if method == "quantile":
        return np.quantile(s, np.arange(1, K) / K)
    sub = s[rng.choice(len(s), min(len(s), 50000), replace=False)]
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(K, n_init=3, random_state=0).fit(sub[:, None])
        grid = np.linspace(s.min(), s.max(), 4000)
        lab = np.argsort(np.argsort(g.means_[:, 0]))[g.predict(grid[:, None])]   # rank of the most probable component
        ch = np.flatnonzero(np.diff(lab) != 0)
        if len(ch) != K - 1 or not np.all(np.diff(lab[np.r_[0, ch + 1]]) > 0):
            return None
        return 0.5 * (grid[ch] + grid[ch + 1])
    if method == "kde_valley":
        from scipy.signal import argrelextrema
        from scipy.stats import gaussian_kde
        grid = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 2000)
        dens = gaussian_kde(sub)(grid)
        mins = argrelextrema(dens, np.less)[0]
        if len(mins) < K - 1:
            return None
        deepest = mins[np.argsort(dens[mins])[: K - 1]]
        return np.sort(grid[deepest])
    raise ValueError(method)
