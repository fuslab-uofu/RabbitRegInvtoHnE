import os
from pathlib import Path
import nibabel as nib
import numpy as np

from RabbitPathFinder import find_all_the_paths
from NewtonAdjointInit import (
    load_moving_vox, load_source_geometry, discover_img_numbers,
    build_b_and_Q, _trilinear_stencil, normalized_adjoint_init, richardson_lucy_solve,
    poisson_i_divergence,
)
from PaintHnEFeatureToInVivo import (
    load_and_aggregate_hne_intensity, paint_feature_volume,
)
from BuildBFromTile224Features import build_b_and_Q_from_tile_features

# Compares three reconstructions of the same feature (a0 vs. the old
# nearest-neighbor voxel-ID pipeline vs. Richardson-Lucy at several iteration
# checkpoints in one run, range-stop logged but not enforced so the full
# trajectory is visible), against either the real registered H&E data or the
# synthetic checkerboard test pattern -- toggle USE_CHECKERBOARD to switch.
# Each data source writes to its own ReconComparison folder so the two never
# collide; NewtonAdjointInit.py's __main__ uses the same toggle/folder
# convention so its Newton-CG output lands alongside this one's.

RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID = "R24-064"
Block = 4
CHECKPOINTS = [1, 2]
CLAMP_OUT_OF_RANGE = True
USE_CHECKERBOARD = False
USE_CELLCOUNT_FEATURE = True  # True: build b/Q from the tile224 bilinear-pull + defect-
                                # threshold pipeline (BuildBFromTile224Features.py) instead
                                # of raw registered-PNG intensity. Ignored when
                                # USE_CHECKERBOARD is True. Was the old voxel-ID nearest-
                                # neighbor pipeline (PaintHnEFeatureToInVivo.build_b_and_Q_
                                # from_feature) -- that pipeline's still available via
                                # load_and_aggregate_hne_intensity/paint_feature_volume
                                # below, kept as a debugging comparison against this b.
FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
DEFECT_THRESHOLD = 0.9  # P(clean) admission cutoff, see BuildBFromTile224Features.py

if USE_CHECKERBOARD:
    data_tag = "checkerboard"
elif USE_CELLCOUNT_FEATURE:
    data_tag = "cellcount"
else:
    data_tag = "realdata"
out_dir = str(Path(__file__).parent / 'TestData' /
              (f"{RabbitID}_Block{Block:02d}_CheckerboardReconComparison" if USE_CHECKERBOARD
               else f"{RabbitID}_Block{Block:02d}_ReconComparison"))
os.makedirs(out_dir, exist_ok=True)
run_tag = f"{RabbitID}_Block{Block:02d}_RLRecon_{data_tag}" + ("_clamped" if CLAMP_OUT_OF_RANGE else "")

if USE_CHECKERBOARD:
    registered_hne_dir = str(Path(__file__).parent / 'TestData' / f"{RabbitID}_Block{Block:02d}_CheckerboardSynthetic")
else:
    registered_hne_dir = os.path.join(RabbitFolder, RabbitID, 'HnE', f"Block{Block:02d}", 'Registered')
cropped_images_dir = find_all_the_paths(RabbitID, Block, RabbitFolder, 'ExVivoBlock')['Fixed_Folder']
cropped_images_dir = os.path.join(cropped_images_dir, 'CroppedImages')

print("Loading h (moving_vox)...")
h, target_affine = load_moving_vox(RabbitID, Block, RabbitFolder, return_target_affine=True)
target_shape = h.shape[:3]
source_shape, source_affine = load_source_geometry(RabbitID, Block, RabbitFolder)

voxel_maps_dir = os.path.join(RabbitFolder, RabbitID, 'InVivo_MR', 'RegDataProc', 'Voxel_Maps')
label_nib_path = os.path.join(voxel_maps_dir, 'voxel_tile_vol.nii.gz')
lookup_path = os.path.join(voxel_maps_dir, 'voxel_tile_vol_id_lookup.npy')
composed_label_path = os.path.join(voxel_maps_dir, f'voxel_tile_vol_ComposedRegToBlock{Block:02d}.nii.gz')

if USE_CELLCOUNT_FEATURE:
    print(f"Building b, Q from tile224 feature '{FEATURE_COL}' (defect threshold={DEFECT_THRESHOLD})...")
    tile224_results, _, _, _ = build_b_and_Q_from_tile_features(
        RabbitID, Block, RabbitFolder, FEATURE_COL, thresholds=(DEFECT_THRESHOLD,))
    b, Q, b_affine = tile224_results[DEFECT_THRESHOLD]
    if b.shape != target_shape:
        raise ValueError(f"tile224 b shape {b.shape} != target_shape {target_shape} from load_moving_vox")
else:
    img_numbers = discover_img_numbers(registered_hne_dir)
    print(f"Building b, Q from {len(img_numbers)} {data_tag} slides...")
    b, Q = build_b_and_Q(registered_hne_dir, cropped_images_dir, target_shape, img_numbers, target_affine)
    b_affine = target_affine

b_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_b_{data_tag}.nii.gz")
nib.save(nib.Nifti1Image(b.astype(np.float32), b_affine), b_path)
print(f"Saved b -> {b_path}")

print("Computing trilinear stencil...")
corners = _trilinear_stencil(h, source_shape)

print("Computing a0 (normalized-adjoint init, same one Newton-CG uses)...")
a0, coverage = normalized_adjoint_init(b, Q, corners, source_shape)
a0_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_a0.nii.gz")
nib.save(nib.Nifti1Image(a0.astype(np.float32), source_affine), a0_path)
print(f"Saved a0 -> {a0_path}")

coverage_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_coverage.nii.gz")
nib.save(nib.Nifti1Image(coverage.astype(np.float32), source_affine), coverage_path)
print(f"Saved coverage -> {coverage_path}")

print(f"Painting the same {data_tag} b through the old nearest-neighbor voxel-ID pipeline...")
feature_series = load_and_aggregate_hne_intensity(b_path, composed_label_path, lookup_path)
painted, painted_affine = paint_feature_volume(label_nib_path, lookup_path, feature_series)
painted_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_painted.nii.gz")
nib.save(nib.Nifti1Image(painted.astype(np.float32), painted_affine), painted_path)
print(f"Saved painted comparison -> {painted_path}")

print(f"\nRunning Richardson-Lucy for {max(CHECKPOINTS)} iterations "
      f"(range-stop logged but not enforced, so all checkpoints get captured)...")
def callback(k, a):
    if k not in CHECKPOINTS:
        return
    out_path = os.path.join(out_dir, f"{run_tag}_iter{k}.nii.gz")
    nib.save(nib.Nifti1Image(a.astype(np.float32), source_affine), out_path)
    div = poisson_i_divergence(a, b, Q, corners)
    print(f"  saved RL iteration {k} -> {out_path}  "
          f"(min={a.min():.3f}, max={a.max():.3f}, I-divergence={div:.6e})", flush=True)


richardson_lucy_solve(b, Q, corners, source_shape, max_iters=max(CHECKPOINTS) + 1,
                       enforce_range_stop=False, clamp_out_of_range=CLAMP_OUT_OF_RANGE,
                       iterate_callback=callback)

print(f"\nDone. All comparison outputs in {out_dir}")
