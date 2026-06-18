Synthetic two-conformation cryoDRGN validation set
==================================================
maps: A=C:\Users\maemm\OneDrive\Desktop\CryoEM\data\maps\J1442_classes\J1442_class06.mrc
      B=C:\Users\maemm\OneDrive\Desktop\CryoEM\data\maps\J1442_classes\J1442_class08.mrc
N=3000 particles (1500 per conformation), D=64, SNR~0.6
NO CTF applied (clean projections + Gaussian noise) -> train WITHOUT --ctf.

Train (CPU, a few minutes):
  cryodrgn train_vae results_cryodrgn/demo_idpose/inputs\particles.mrcs --poses results_cryodrgn/demo_idpose/inputs\poses.pkl --zdim 4 -n 20 --enc-dim 128 --enc-layers 2 --dec-dim 128 --dec-layers 2 --uninvert-data=False -o results_cryodrgn/demo_idpose\train

Analyze:
  cryodrgn analyze results_cryodrgn/demo_idpose\train 19

Compare latent clusters to ground truth:
  python scripts/compare_cryodrgn_classification.py \
    --workdir results_cryodrgn/demo_idpose\train --epoch 19 \
    --gt-labels results_cryodrgn/demo_idpose/inputs\gt_labels.pkl --k 2 \
    -o results_cryodrgn/demo_idpose\comparison
