#!/usr/bin/env python
"""Masked half-map FSC resolution (fastbp, 256 px) of exported groups and of their size- and view-matched
random controls: tells 'these particles reconstruct worse than random ones' from 'ab initio / NU failed on this subset'.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np

from fastbp import FastBP, fsc_band, fsc_curve, resolution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True, help="256-px soft mask .npy")
    p.add_argument("--export", nargs="+", required=True, help="TAG=dir with *_ind.pkl")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--random-sizes", type=int, nargs="*", default=[10000, 15000, 20000, 30000, 40000, 60000, 100000])
    p.add_argument("-o", "--out", type=Path, required=True)
    a = p.parse_args()
    bp = FastBP(str(a.inputs / "particles.256.mrcs"), str(a.inputs / "poses.pkl"), str(a.inputs / "ctf.pkl"))
    mask = np.load(a.mask)
    n = mask.shape[0]

    def res(idx):
        out = []
        for s in range(a.seeds):
            h = bp.halves(idx, s)
            c = fsc_curve(h[0], h[1], mask)
            out.append(dict(res=resolution(c, n, bp.apix), band=fsc_band(c, n, bp.apix), curve=np.round(c, 4).tolist()))
        return out

    out = dict(random={}, groups={})
    N = bp.data.N
    for k in a.random_sizes:  # size -> resolution curve for random subsets
        out["random"][k] = [res(np.random.default_rng(100 + r).choice(N, k, replace=False)) for r in range(2)]
        print("random", k, [[(round(x["res"], 2), round(x["band"], 3)) for x in rr] for rr in out["random"][k]], flush=True)
    for spec in a.export:
        tag, d = spec.split("=", 1)
        for f in sorted(glob.glob(f"{d}/*_ind.pkl")):
            name = os.path.basename(f)[:-8]
            if "__" in name:
                continue
            g = dict(n=None)
            for suf in ("", "__control_a", "__control_b"):
                ff = f"{d}/{name}{suf}_ind.pkl"
                if not os.path.exists(ff):
                    continue
                idx = np.asarray(pickle.load(open(ff, "rb")))
                g["n" if not suf else f"n{suf}"] = int(len(idx))
                g[f"res{suf}"] = res(idx)
            out["groups"][f"{tag}:{name}"] = g
            print(tag, name, {k: ([(round(x["res"], 2), round(x["band"], 3)) for x in v] if isinstance(v, list) else v)
                              for k, v in g.items()}, flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
