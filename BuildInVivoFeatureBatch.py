import os

import nibabel as nib
import numpy as np
import pandas as pd

from BuildBFromTile224Features import build_b_and_Q_from_tile_features
from NewtonAdjointInit import normalized_adjoint_init, richardson_lucy_solve

# Production batch pipeline: tile224 bilinear-pull + defect-threshold b/Q
# (BuildBFromTile224Features.py) -> normalized-adjoint a0/coverage -> one
# Richardson-Lucy iteration, for every rabbit/block in READY_RABBITS_CSV.
#
# This supersedes RichardsonLucyTest.py / BuildBFromTile224Features.py's own
# __main__ as the production path now that threshold=0.9 + binary Q are
# settled -- those two still have the checkerboard/old-nearest-neighbor/
# multi-threshold debugging scaffolding, kept as-is for reference, just not
# carried into this script.

RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
READY_RABBITS_CSV = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/ReadyRabbits.csv'
FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
DEFECT_THRESHOLD = 0.9  # P(clean) admission cutoff, see BuildBFromTile224Features.py
CLAMP_OUT_OF_RANGE = True


def build_invivo_feature_volume(RabbitID, Block, out_dir):
    """b, Q (tile224 + defect threshold) -> a0 -> one Richardson-Lucy
    iteration, all saved to out_dir. Returns the three output paths."""
    print(f"Building b, Q from tile224 feature '{FEATURE_COL}' (defect threshold={DEFECT_THRESHOLD})...")
    results, corners, source_shape, source_affine = build_b_and_Q_from_tile_features(
        RabbitID, Block, RabbitFolder, FEATURE_COL, thresholds=(DEFECT_THRESHOLD,))
    b, Q, b_affine = results[DEFECT_THRESHOLD]

    b_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_b_{FEATURE_COL}.nii.gz")
    nib.save(nib.Nifti1Image(b.astype(np.float32), b_affine), b_path)
    print(f"Saved b -> {b_path}")

    print("Computing a0 (normalized-adjoint init)...")
    a0, _ = normalized_adjoint_init(b, Q, corners, source_shape)
    a0_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_a0.nii.gz")
    nib.save(nib.Nifti1Image(a0.astype(np.float32), source_affine), a0_path)
    print(f"Saved a0 -> {a0_path}")

    print("Running 1 Richardson-Lucy iteration...")
    a_rl1 = richardson_lucy_solve(b, Q, corners, source_shape, max_iters=1,
                                   enforce_range_stop=False, clamp_out_of_range=CLAMP_OUT_OF_RANGE)
    rl_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_RL_iter1.nii.gz")
    nib.save(nib.Nifti1Image(a_rl1.astype(np.float32), source_affine), rl_path)
    print(f"Saved RL iter1 -> {rl_path}")

    return b_path, a0_path, rl_path


if __name__ == '__main__':
    ready_rabbits = pd.read_csv(READY_RABBITS_CSV)

    for _, row in ready_rabbits.iterrows():
        RabbitID, Block = row['rabbit'], int(row['block'])
        print(f"\n=== {RabbitID} Block{Block:02d} ===")
        out_dir = os.path.join(RabbitFolder, RabbitID, 'Analysis', f'Block{Block:02d}')
        os.makedirs(out_dir, exist_ok=True)
        try:
            build_invivo_feature_volume(RabbitID, Block, out_dir)
        except Exception as exc:
            print(f"[{RabbitID} Block{Block:02d}] FAILED: {exc}")

    print("\nDone.")
