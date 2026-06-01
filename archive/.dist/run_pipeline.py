"""
Headless runner for the CryoSPARC analysis scripts.

- Forces the Agg matplotlib backend (no GUI).
- Monkeypatches builtins.input() to feed the .cs file path.
- Monkeypatches plt.show() to save each figure as PNG into an output dir.
- Suppresses sns.pairplot blocking by also catching Figure-level saves.

Usage:
    python run_pipeline.py <script.py> <cs_file> <out_dir>
"""
import builtins
import os
import runpy
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_path = sys.argv[1]
cs_path = sys.argv[2]
out_dir = sys.argv[3]
os.makedirs(out_dir, exist_ok=True)

# feed the input() prompt
_inputs = iter([cs_path])
builtins.input = lambda *a, **k: next(_inputs)

# save every plt.show() call
_counter = {"n": 0}
_orig_show = plt.show
def _save_show(*args, **kwargs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _counter["n"] += 1
        fname = os.path.join(out_dir, f"fig_{_counter['n']:03d}.png")
        try:
            fig.savefig(fname, dpi=80, bbox_inches="tight")
        except Exception as e:
            print(f"  [save error for {fname}: {e}]")
        plt.close(fig)
plt.show = _save_show

print(f"=== Running {script_path} on {cs_path} ===")
try:
    runpy.run_path(script_path, run_name="__main__")
    print(f"=== DONE: {_counter['n']} figures saved to {out_dir} ===")
except SystemExit as e:
    print(f"=== SystemExit({e.code}): {_counter['n']} figures saved ===")
except Exception:
    traceback.print_exc()
    print(f"=== FAILED after {_counter['n']} figures (saved to {out_dir}) ===")
    sys.exit(1)
