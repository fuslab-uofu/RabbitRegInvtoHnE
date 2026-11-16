import os

import nibabel as nib
import numpy as np
import pandas as pd

from BuildBFromTile224Features import build_b_and_Q_from_tile_features
from NewtonAdjointInit import normalized_adjoint_init, richardson_lucy_solve, load_native_geometry

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
READY_RABBITS_CSV = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/ReadyRabbitsNew.csv'
FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
DEFECT_THRESHOLD = 0.9  # P(clean) admission cutoff, see BuildBFromTile224Features.py
CLAMP_OUT_OF_RANGE = True
TAU = 1e-6  # must match normalized_adjoint_init/richardson_lucy_solve's own default


def _reorient_to_native(arr, canonical_affine, native_shape, native_affine):
    """Reorient a (canonical-space) array + affine pair back to the same
    scan's raw on-disk orientation -- the convention Day3End_Registered and
    the rest of the MR pipeline already use, so this output lands
    voxel-for-voxel aligned with it. See load_native_geometry."""
    canonical_img = nib.Nifti1Image(arr, canonical_affine)
    ornt = nib.orientations.ornt_transform(
        nib.io_orientation(canonical_affine),
        nib.io_orientation(native_affine),
    )
    reoriented = canonical_img.as_reoriented(ornt)
    assert reoriented.shape[:3] == native_shape, (
        f"reoriented shape {reoriented.shape[:3]} != native shape {native_shape}")
    return np.asarray(reoriented.dataobj), reoriented.affine


def build_invivo_feature_volume(RabbitID, Block, out_dir):
    """b, Q (tile224 + defect threshold) -> a0 -> one Richardson-Lucy
    iteration, all saved to out_dir. Returns the three output paths."""
    print(f"Building b, Q from tile224 feature '{FEATURE_COL}' (defect threshold={DEFECT_THRESHOLD})...")
    results, corners, source_shape, source_affine = build_b_and_Q_from_tile_features(
        RabbitID, Block, RabbitFolder, [FEATURE_COL], thresholds=(DEFECT_THRESHOLD,))
    b, Q, b_affine = results[(FEATURE_COL, DEFECT_THRESHOLD)]

    b_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_b_{FEATURE_COL}.nii.gz")
    nib.save(nib.Nifti1Image(b.astype(np.float32), b_affine), b_path)
    print(f"Saved b -> {b_path}")

    native_shape, native_affine = load_native_geometry(RabbitID, Block, RabbitFolder)

    print("Computing a0 (normalized-adjoint init)...")
    a0, coverage = normalized_adjoint_init(b, Q, corners, source_shape)
    # Uncovered voxels get a fixed NaN sentinel (rather than a0's own 0.0
    # fallback, or RL's data-minimum floor below) -- one unambiguous
    # no-coverage value across both outputs and every rabbit, matching the
    # NaN convention ExtractMRFeatures.py/valid_mask() already expect.
    covered = coverage > TAU
    a0 = np.where(covered, a0, np.nan)
    a0_native, a0_affine = _reorient_to_native(a0.astype(np.float32), source_affine,
                                                native_shape, native_affine)
    a0_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_a0.nii.gz")
    nib.save(nib.Nifti1Image(a0_native, a0_affine), a0_path)
    print(f"Saved a0 -> {a0_path}")

    print("Running 1 Richardson-Lucy iteration...")
    a_rl1 = richardson_lucy_solve(b, Q, corners, source_shape, max_iters=1,
                                   enforce_range_stop=False, clamp_out_of_range=CLAMP_OUT_OF_RANGE)
    # coverage/covered depend only on Q/corners/source_shape, not on b or the
    # a0 fallback, so the same mask from the a0 call above applies here too.
    a_rl1 = np.where(covered, a_rl1, np.nan)
    a_rl1_native, rl_affine = _reorient_to_native(a_rl1.astype(np.float32), source_affine,
                                                   native_shape, native_affine)
    rl_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_RL_iter1.nii.gz")
    nib.save(nib.Nifti1Image(a_rl1_native, rl_affine), rl_path)
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
