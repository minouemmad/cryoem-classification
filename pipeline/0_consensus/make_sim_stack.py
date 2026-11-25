"""Homogeneous synthetic stack for a dataset (null for the VAE + recon-tree groups; recipe of J4001_gP25_WT/sim_homog).
FastBP.simulate on RES/NAME/inputs: image_i = a_i * CTF_i * slice(consensus, pose_i) + Gaussian noise with the per-shell residual
spectrum (same poses, CTFs, scales, noise level as the data, no heterogeneity). Real-space images, centred (poses.pkl gets zero
translations), sign chosen so cryoDRGN's --invert-data default reads back the same Hartley image. -> RES/NAME/sim_homog/inputs
Usage: make_sim_stack.py NAME [SEED]"""
import os, pickle, sys
import numpy as np, torch, mrcfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastbp import FastBP  # noqa: E402
from cryodrgn import fft  # noqa: E402
from registry import R  # noqa: E402
name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261106
I = f"{R}/{name}/inputs"; O = f"{R}/{name}/sim_homog/inputs"; os.makedirs(O, exist_ok=True)
bp = FastBP(f"{I}/particles.128.mrcs", f"{I}/poses.pkl", f"{I}/ctf.pkl"); N, D = bp.data.N, bp.D
bp.simulate(seed=seed)
rot, trans = pickle.load(open(f"{I}/poses.pkl", "rb"))
pickle.dump((rot, np.zeros_like(trans)), open(f"{O}/poses.pkl", "wb"))
if not os.path.exists(f"{O}/ctf.pkl"):
    os.symlink(f"{I}/ctf.pkl", f"{O}/ctf.pkl")
out = np.lib.format.open_memmap(f"{O}/tmp_stack.npy", mode="w+", dtype=np.float32, shape=(N, D - 1, D - 1))
for s in range(0, N, 2000):
    ii = np.arange(s, min(N, s + 2000)); H = torch.zeros((len(ii), D * D), device=bp.dev)
    H[:, bp.mask] = bp.sim[torch.as_tensor(ii, device=bp.dev)].float()
    out[s: s + len(ii)] = (-fft.iht2_center(H.view(-1, D, D)[:, :-1, :-1])).cpu().numpy()
out.flush()
with mrcfile.new(f"{O}/particles.128.mrcs", overwrite=True) as m:
    m.set_data(out); m.voxel_size = bp.apix * 1.0
os.remove(f"{O}/tmp_stack.npy")
sim0 = bp.sim[:2000].float().clone(); del bp; torch.cuda.empty_cache()
b2 = FastBP(f"{O}/particles.128.mrcs", f"{O}/poses.pkl", f"{O}/ctf.pkl")
ff, _, _ = b2._batch(np.arange(2000))
r = torch.nn.functional.cosine_similarity(ff, sim0, dim=1)
print(f"round trip cos: min {r.min().item():.5f} median {r.median().item():.5f}; rel err {((ff - sim0).norm() / sim0.norm()).item():.2e}", flush=True)
print("sim stack done", flush=True)
