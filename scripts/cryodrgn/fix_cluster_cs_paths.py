#!/usr/bin/env python
"""Rewrite relative CryoSPARC path fields in exported cryoDRGN cluster .cs files
to absolute paths so they resolve on the cluster.

The exported cryoDRGN cluster .cs files store project-relative paths in their
path fields (blob/path, motion/path, location/micrograph_path), e.g.
    J2694/reconstructed/<uid>_particles.mrc
These reference CryoSPARC jobs that live in the project directory. When the
cluster .cs files are placed in
    <PROJECT>/cryodrgn/
their source job directories (J76, J2694, J2014, ...) are one level up, in
    <PROJECT>/
so the relative paths must be prefixed with the absolute project base for
CryoSPARC to find the particle stacks.

Usage (from the workspace root):
    python scripts/cryodrgn/fix_cluster_cs_paths.py \
        --base /mnt/disk1/cryoSPARCWorkingDirectory/18may31h/P5_fromEast \
        results_cryodrgn/J207/gmm_full_latent_k6 \
        results_cryodrgn/J2708/gmm_full_latent_k4 \
        results_cryodrgn/J4624/gmm_full_latent_k5

Only values that are NOT already absolute (do not start with '/') are rewritten,
so the script is idempotent. A one-time backup (<name>.cs.orig) is written for
every file that is modified unless --no-backup is given.
"""
import argparse
import glob
import os
import shutil

import numpy as np


def path_fields(dtype):
    """Return the names of string fields that hold CryoSPARC file paths."""
    return [n for n in dtype.names if n.endswith("path")]


def rewrite_file(cs_path, base, backup=True, dry_run=False):
    data = np.load(cs_path)
    fields = path_fields(data.dtype)
    if not fields:
        return cs_path, {}, False

    base = base.rstrip("/")
    changed = {}
    modified = False
    # Collect rewritten columns; fixed-width string fields must be widened or the
    # longer absolute paths get silently truncated.
    new_columns = {}
    new_dtype = []
    for name in data.dtype.names:
        col = data[name]
        if name not in fields:
            # data.dtype[name] already carries any subarray shape.
            new_dtype.append((name, data.dtype[name]))
            continue
        is_bytes = col.dtype.kind == "S"
        n_rewritten = 0
        new_vals = []
        for v in col:
            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
            if s and not s.startswith("/"):
                s = base + "/" + s
                n_rewritten += 1
            new_vals.append(s.encode() if is_bytes else s)
        if n_rewritten:
            changed[name] = n_rewritten
            modified = True
        arr = np.array(new_vals, dtype=col.dtype.kind + str(
            max(int(col.dtype.itemsize // (1 if is_bytes else 4)),
                max((len(x) for x in new_vals), default=1))))
        new_columns[name] = arr
        new_dtype.append((name, arr.dtype))

    if modified and not dry_run:
        out = np.empty(data.shape, dtype=np.dtype(new_dtype))
        for name in data.dtype.names:
            out[name] = new_columns[name] if name in new_columns else data[name]
        data = out
        if backup:
            bak = cs_path + ".orig"
            if not os.path.exists(bak):
                shutil.copy2(cs_path, bak)
        np.save(cs_path, data)
        # np.save appends .npy; restore the .cs name.
        if os.path.exists(cs_path + ".npy"):
            os.replace(cs_path + ".npy", cs_path)

    return cs_path, changed, modified


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Directories and/or .cs files to fix.")
    ap.add_argument("--base", required=True,
                    help="Absolute project base to prepend (the directory that "
                         "contains the source job dirs, e.g. "
                         "/mnt/disk1/cryoSPARCWorkingDirectory/18may31h/P5_fromEast).")
    ap.add_argument("--glob", default="*_cluster_c*.cs",
                    help="Glob used to find .cs files inside directories "
                         "(default: %(default)s).")
    ap.add_argument("--no-backup", action="store_true",
                    help="Do not write <name>.cs.orig backups.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, args.glob))))
        else:
            files.append(p)

    if not files:
        raise SystemExit("No .cs files matched.")

    total = 0
    for f in files:
        _, changed, modified = rewrite_file(
            f, args.base, backup=not args.no_backup, dry_run=args.dry_run)
        tag = "DRY-RUN" if args.dry_run else ("updated" if modified else "no change")
        detail = ", ".join(f"{k}:{v}" for k, v in changed.items()) or "-"
        print(f"[{tag}] {f}  ({detail})")
        total += int(modified)

    verb = "would be updated" if args.dry_run else "updated"
    print(f"\n{total}/{len(files)} files {verb}.")


if __name__ == "__main__":
    main()
