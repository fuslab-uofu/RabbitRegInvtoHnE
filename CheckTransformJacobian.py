import os
import shutil
import tempfile
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, map_coordinates
from RabbitPathFinder import find_all_the_paths

# Same fragmented-vs-clean comparison as CheckWarpJacobian.py, but evaluating the
# actual registration transforms (Slicer .h5 + .pt field, all three stages) directly,
# instead of inferring warp behavior from the already-resampled, potentially-aliased
# label volume.
#
# No inversion of anything is needed: this evaluates the Jacobian of the native
# fixed->moving ("pull") direction each stage's transform/.pt field already supports
# without inversion (exactly the direction ApplySlicerTransform/ApplyDfield use). A
# function and its inverse have matrix-inverse Jacobians at corresponding points, and
# a matrix and its inverse always share the same determinant sign — so folding
# (negative determinant) is detectable in this direction just as validly as in the
# true forward (InVivo->BlockFace) direction, which would require inverting the .pt
# fields via the same fixed-point iteration we already know is unreliable near
# out-of-FOV regions.

RABBIT_FOLDER = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RABBIT_ID = 'R24-058'
BLOCK = 6
Block = f'Block{BLOCK:02d}'
PAD_VOXELS = 20
N_SAMPLE = 200
FD_EPS_MM = 0.01  # finite-difference step for the numerical Jacobian, physical mm

PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]
# Order pull_bf_to_invivo actually walks stages in (reversed(stages), BlockFace-end first).
STAGE_LABELS_REVERSED = ['fixed=BlockFace (ExVivoBlock->BlockFace)',
                          'fixed=ExVivoBlock (ExVivo->ExVivoBlock)',
                          'fixed=ExVivo (InVivo->ExVivo)']

rabbase = os.path.join(RABBIT_FOLDER, RABBIT_ID)
voxel_maps_dir = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps')
bf_label_path = os.path.join(voxel_maps_dir, f'voxel_tile_vol_unique_pad{PAD_VOXELS}_ComposedRegTo{Block}.nii.gz')
geoms_dir = os.path.join(rabbase, 'Analysis', Block)


def _load_transform(path):
    """Same temp-copy pattern used elsewhere to sidestep HDF5 file locking."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        return sitk.ReadTransform(tmp_path)
    finally:
        os.unlink(tmp_path)


def _load_stage(moving_key):
    """Everything needed to pull a point from this stage's FIXED physical space back
    to its MOVING physical space, in the same native (fixed->moving) direction
    ApplySlicerTransform/ApplyDfield already use — no inversion required. .TransformPoint()
    works transparently whether the .h5 is a bare affine or a composite with an
    embedded displacement field, so no manual decomposition is needed here either."""
    paths = find_all_the_paths(RABBIT_ID, BLOCK, RABBIT_FOLDER, moving_key)
    fixed_nib_canonical = nib.as_closest_canonical(nib.load(str(paths['Fixed_FilePath'])))
    slicer_path = next(p for p in Path(paths['RegFold']).glob('*.h5') if not p.name.startswith('._'))
    dfield_path = next(p for p in Path(paths['RegFold']).glob('*.pt') if not p.name.startswith('._'))
    transform = _load_transform(slicer_path)
    dfield = torch.load(str(dfield_path), map_location='cpu')['data'].numpy()  # (3, X, Y, Z)
    X, Y, Z = fixed_nib_canonical.shape[:3]
    dfield = dfield[:, :X, :Y, :Z]
    return fixed_nib_canonical, transform, dfield


def _dfield_pull_vox(dfield, vox_xyz):
    """channels [2]=X,[1]=Y,[0]=Z, subtract 1 for 0-indexed output. dfield's spatial
    axes are (X,Y,Z), same order as vox_xyz -- no axis reversal before sampling (see
    ComposedTransformReg.py, which had this exact spurious-reversal bug and was fixed
    against the validated ApplyTransforms.py:163 reference). Also reports whether
    vox_xyz fell outside the field's own array bounds: map_coordinates(mode='nearest')
    silently clamps to the edge value in that case, which is a real gap in the sample
    (not a legitimate field value) -- same issue ComposedTransformReg.py's invalid-mask
    fix addresses for the forward resample."""
    X, Y, Z = dfield.shape[1:]
    oob = not (0 <= vox_xyz[0] <= X - 1 and 0 <= vox_xyz[1] <= Y - 1 and 0 <= vox_xyz[2] <= Z - 1)
    coords = np.array(vox_xyz).reshape(3, 1)  # x,y,z for map_coordinates, matches dfield's own axis order
    fx = map_coordinates(dfield[2], coords, order=1, mode='nearest')[0] - 1
    fy = map_coordinates(dfield[1], coords, order=1, mode='nearest')[0] - 1
    fz = map_coordinates(dfield[0], coords, order=1, mode='nearest')[0] - 1
    return np.array([fx, fy, fz]), oob


def _stage_pull_point(point_ras_fixed, fixed_nib_canonical, dfield, transform):
    """One stage's fixed-physical -> moving-physical point pull, matching how
    ApplySlicerTransform (Slicer transform) then ApplyDfield (deformation field)
    actually resample content: the dfield operates first (within the fixed grid),
    then the Slicer transform carries the result into the moving image's physical
    space — same order MultiStepReg applies them in."""
    hom = np.append(point_ras_fixed, 1.0)
    vox_fixed = (np.linalg.inv(fixed_nib_canonical.affine) @ hom)[:3]
    vox_intermediate, oob = _dfield_pull_vox(dfield, vox_fixed)
    hom2 = np.append(vox_intermediate, 1.0)
    point_ras_intermediate = (fixed_nib_canonical.affine @ hom2)[:3]
    point_lps_intermediate = point_ras_intermediate * np.array([-1., -1., 1.])
    point_lps_moving = np.array(transform.TransformPoint(point_lps_intermediate.tolist()))
    return point_lps_moving * np.array([-1., -1., 1.]), oob


def pull_bf_to_invivo(point_ras_bf, stages):
    """stages: [(fixed_nib_canonical, transform, dfield), ...] for InVivo->ExVivo,
    ExVivo->ExVivoBlock, ExVivoBlock->BlockFace, in that (PROGRESSION/forward) order.
    Walked in reverse here to pull a BlockFace point back to InVivo. Returns
    (point, oob_by_stage), oob_by_stage a 3-bool list in the order stages are actually
    walked (STAGE_LABELS_REVERSED) so callers can tell which stage is responsible,
    instead of only a single collapsed flag."""
    p = point_ras_bf
    oob_by_stage = []
    for fixed_nib_canonical, transform, dfield in reversed(stages):
        p, oob = _stage_pull_point(p, fixed_nib_canonical, dfield, transform)
        oob_by_stage.append(oob)
    return p, oob_by_stage


def jacobian_at(point_ras_bf, stages, eps=FD_EPS_MM):
    """Also returns centroid_oob_by_stage (the exact, unperturbed point) separately
    from probe_oob_by_stage (any of the 6 +/-eps finite-difference neighbors) -- a
    centroid coming from TilestoVoxelsV2.py's traced tissue should already be a real,
    successfully-registered point (ComposedReg zero-fills anything invalid before it
    could ever be traced), so centroid OOB is a red flag; probe-only OOB just means a
    +/-eps neighbor stepped past a real nearby field/crop boundary."""
    _, centroid_oob_by_stage = pull_bf_to_invivo(point_ras_bf, stages)
    cols = []
    probe_oob_by_stage = [False, False, False]
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = eps
        p_plus, oob_plus = pull_bf_to_invivo(point_ras_bf + d, stages)
        p_minus, oob_minus = pull_bf_to_invivo(point_ras_bf - d, stages)
        probe_oob_by_stage = [a or b or c for a, b, c in zip(probe_oob_by_stage, oob_plus, oob_minus)]
        cols.append((p_plus - p_minus) / (2 * eps))
    return np.stack(cols, axis=1), centroid_oob_by_stage, probe_oob_by_stage


def bf_centroid_vox(vid, obj_slices, bf_labels):
    sl = obj_slices[vid - 1]
    if sl is None:
        return None
    local_coords = np.argwhere(bf_labels[sl] == vid)
    offset = np.array([s.start for s in sl])
    return local_coords.mean(axis=0) + offset


if __name__ == '__main__':
    print("Loading BlockFace label volume...")
    bf_nib = nib.load(bf_label_path)
    bf_labels = bf_nib.get_fdata().astype(np.int64)

    print("Finding each label's bounding box (single pass)...")
    obj_slices = find_objects(bf_labels)

    # Voxel IDs pre-classified by TilestoVoxelsV2.py: every ID flagged as split
    # across some 2D H&E cross-section, split further into ones still disconnected
    # in true 3D vs. ones that turned out to be one connected piece (a slicing-
    # plane artifact). Using this pool instead of a fresh from-scratch full-volume
    # connectivity scan isolates real 3D disconnection as the only difference
    # between the two groups -- both were already flagged in 2D, so this controls
    # for confounds (size/shape/location) that come along with "was flagged at all".
    disconnected_ids = np.load(os.path.join(geoms_dir, f'disconnected_in_3d_ids_pad{PAD_VOXELS}.npy')).tolist()
    connected_ids = np.load(os.path.join(geoms_dir, f'connected_in_3d_ids_pad{PAD_VOXELS}.npy')).tolist()
    print(f"Loaded {len(disconnected_ids)} disconnected-in-3D and {len(connected_ids)} connected-in-3D "
          f"voxel IDs (TilestoVoxelsV2.py's 2D-flagged pool)")

    print("Loading the three registration stages (Slicer transform + .pt field each)...")
    stages = [_load_stage(moving_key) for moving_key in PROGRESSION[:3]]  # InVivo, ExVivo, ExVivoBlock

    random.seed(0)
    sample_frag = disconnected_ids
    sample_clean = random.sample(connected_ids, min(len(disconnected_ids), len(connected_ids)))

    def summarize(ids, label):
        dets = []
        tainted_dets = []
        clean_dets = []
        n_centroid_oob = 0
        n_probe_oob = 0
        n_exact_zero = 0
        centroid_oob_stage_counts = [0, 0, 0]
        probe_oob_stage_counts = [0, 0, 0]
        for i, vid in enumerate(ids):
            vox = bf_centroid_vox(vid, obj_slices, bf_labels)
            if vox is None:
                continue
            hom = np.append(vox, 1.0)
            point_ras_bf = (bf_nib.affine @ hom)[:3]
            J, centroid_oob_by_stage, probe_oob_by_stage = jacobian_at(point_ras_bf, stages)
            det = np.linalg.det(J)
            dets.append(det)
            tainted = any(probe_oob_by_stage)
            (tainted_dets if tainted else clean_dets).append(det)
            if any(centroid_oob_by_stage):
                n_centroid_oob += 1
                for s, flag in enumerate(centroid_oob_by_stage):
                    centroid_oob_stage_counts[s] += int(flag)
            if tainted:
                n_probe_oob += 1
                for s, flag in enumerate(probe_oob_by_stage):
                    probe_oob_stage_counts[s] += int(flag)
            if det == 0.0:
                n_exact_zero += 1
            if (i + 1) % 50 == 0:
                print(f"  {label}: {i + 1}/{len(ids)}...")
        dets = np.array(dets)
        tainted_dets = np.array(tainted_dets)
        clean_dets = np.array(clean_dets)
        print(f"{label}: n={len(dets)}, median|det|={np.median(np.abs(dets)):.4f}, "
              f"frac negative (folded) = {np.mean(dets < 0):.3f}, "
              f"exact-zero det = {n_exact_zero}/{len(dets)}")
        print(f"  centroid OOB (should be ~0, these are real traced tissue points) = "
              f"{n_centroid_oob}/{len(dets)} -- by stage: "
              f"{dict(zip(STAGE_LABELS_REVERSED, centroid_oob_stage_counts))}")
        print(f"  probe-only OOB (a +/-eps FD neighbor crossed a real field/crop edge) = "
              f"{n_probe_oob}/{len(dets)} -- by stage: "
              f"{dict(zip(STAGE_LABELS_REVERSED, probe_oob_stage_counts))}")
        if len(clean_dets):
            print(f"  clean subset (no OOB taint): n={len(clean_dets)}, "
                  f"frac negative = {np.mean(clean_dets < 0):.3f}")
        if len(tainted_dets):
            print(f"  OOB-tainted subset (this is where a masked fold would show up): "
                  f"n={len(tainted_dets)}, frac negative = {np.mean(tainted_dets < 0):.3f}, "
                  f"det range = [{tainted_dets.min():.3f}, {tainted_dets.max():.3f}]")
        return dets

    frag_dets = summarize(sample_frag, "Disconnected-in-3D")
    clean_dets = summarize(sample_clean, "Connected-in-3D")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(frag_dets.min(), clean_dets.min()), max(frag_dets.max(), clean_dets.max()), 60
    )
    ax.hist(clean_dets, bins=bins, alpha=0.6, label=f'Connected-in-3D (n={len(clean_dets)})', color='tab:blue')
    ax.hist(frag_dets, bins=bins, alpha=0.6, label=f'Disconnected-in-3D (n={len(frag_dets)})', color='tab:red')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Local Jacobian determinant of the actual composed transform chain')
    ax.set_ylabel('Count')
    ax.set_title('Transform-based Jacobian at disconnected-in-3D vs. connected-in-3D voxel locations')
    ax.legend()
    plt.tight_layout()
    plt.show()
