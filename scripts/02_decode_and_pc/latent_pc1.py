"""PCA in LATENT space: PC scores of a model's z.<EPOCH>.pkl (all particles), saved like volume_pc_scores.py output (N x 10, column 0 = PC1) so that residual_fit.py can take them as MCR_SCORES. latent_pc"""
import pickle, sys
import numpy as np
z = pickle.load(open(f'{sys.argv[1]}/z.{sys.argv[2]}.pkl', 'rb')); z = np.asarray(z, np.float64); z -= z.mean(0); u, s, vt = np.linalg.svd(z, full_matrices=False); sc = (z @ vt.T)[:, :10].astype(np.float32)
np.save(sys.argv[3], sc); print('latent PCA', z.shape, 'explained variance', np.round(s[:4] ** 2 / (s ** 2).sum(), 3), '->', sys.argv[3])
