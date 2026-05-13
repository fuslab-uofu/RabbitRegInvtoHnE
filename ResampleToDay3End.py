"""
ResampleToDay3End.py

Resample Day0 and Day3_start volumes into Day3_end voxel space.

Two transform chains:
  Day3_start vols  →  [Slicer inv composite]           →  Day3_end
  Day0 vols        →  [Slicer inv composite + F^{-1}]  →  Day3_end

The Day3_end → Day0 combined displacement field is cached as a .npy so it only
needs to be computed once per rabbit.

Per-rabbit Day3 filenames are read from {DAY3_DIR}/{RABBIT_ID_underscored}_InVDay3Log.csv.
CSV columns: 'File Name' (no extension), 'Home' (Fixed | End | Start).
"""

import os
import glob
import shutil
import tempfile
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
import csv

from ApplyTransforms import ApplySlicerTransform, propagate_tiles_to_day0

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------
# Paths — edit these two lines per rabbit
# ---------------------------------------------------------------------------
RABBIT_FOLDER = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RABBIT_ID     = 'R23-055'

_base     = os.path.join(RABBIT_FOLDER, RABBIT_ID, 'InVivo_MR', 'InVMRDataSets')
DAY3_DIR  = os.path.join(_base, 'Day3')
DAY0_DIR  = os.path.join(_base, 'Day0')
XFMS_DIR  = os.path.join(_base, 'Transforms')
OUT_DIR   = os.path.join(_base, 'Day3End_Registered')

# ---------------------------------------------------------------------------
# Load Day3 filenames from log CSV
# ---------------------------------------------------------------------------
def _load_day3_log(day3_dir, rabbit_id):
    log_name = rabbit_id.replace('-', '_') + '_InVDay3Log.csv'
    log_path = os.path.join(day3_dir, log_name)
    roles = {}
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            roles[row['Home'].strip()] = os.path.join(day3_dir, row['File Name'].strip() + '.nii.gz')
    return roles

_day3 = _load_day3_log(DAY3_DIR, RABBIT_ID)

DAY3_END_PATH     = _day3['End']
DAY3_START_STRUCT = _day3['Fixed']
DAY3_START_EXTRA  = [v for k, v in _day3.items() if k == 'Start']

SLICER_INV_CACHE  = os.path.join(DAY3_DIR, 'Day3_end_to_start_inv_cached.h5')
AFFINE_FIELD_PATH = os.path.join(XFMS_DIR, 'Affine_deformation.npy')
SPLINES_FIELD_PATH= os.path.join(XFMS_DIR, 'SplinesProjection.npy')
SDIFF_FIELD_PATH  = os.path.join(XFMS_DIR, 'sdiff.npy')
DISPLACEMENT_CACHE= os.path.join(OUT_DIR, 'day3end_to_day0_voxel_coords.npy')

# Increase if Day0→Day3_end still looks warped in large-deformation regions.
# Delete the cached .npy and re-run after changing this.
N_ITER = 60   # more iterations needed with damping
ALPHA  = 0.25  # <1.0 damps oscillation in large-deformation areas; use 1.0 for undamped

# 'fixedpoint' : iterative inversion (current approach)
# 'sitk'       : compose F forward then use sitk.InvertDisplacementField (more robust)
APS_INV_METHOD = 'fixedpoint'

# Interpolation order for resampling: 1=linear, 0=nearest neighbour
INTERP_ORDER = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_transform(path):
    """Load a Slicer .h5 transform via temp copy to sidestep HDF5 locking."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        return sitk.ReadTransform(tmp_path)
    finally:
        os.unlink(tmp_path)


def _apply_slicer_inv_vectorized(pts_ras, inv_composite):
    """
    Apply the cached Slicer inverse composite (Day3_end → Day3_start) to an
    (N, 3) array of physical RAS points, fully vectorized via map_coordinates.

    The composite applies inv_disp first, then inv_affine (both in LPS).
    Returns (N, 3) physical RAS points in Day3_start space.
    """
    pts_lps = pts_ras * np.array([-1., -1., 1.])

    # --- inv_disp (DisplacementFieldTransform) ---
    inv_disp_t = sitk.DisplacementFieldTransform(inv_composite.GetNthTransform(0))
    disp_img   = inv_disp_t.GetDisplacementField()
    disp_arr   = sitk.GetArrayFromImage(disp_img)  # (Z, Y, X, 3) LPS displacements

    origin    = np.array(disp_img.GetOrigin())              # LPS (X, Y, Z)
    spacing   = np.array(disp_img.GetSpacing())             # (X, Y, Z)
    direction = np.array(disp_img.GetDirection()).reshape(3, 3)  # LPS direction cosines

    # LPS physical → field voxel (X, Y, Z), accounting for direction matrix
    # voxel = D^{-1} S^{-1} (p - origin), where D=direction, S=diag(spacing)
    phys_to_vox = np.linalg.inv(direction @ np.diag(spacing))
    field_vox   = ((pts_lps - origin) @ phys_to_vox.T)    # (N, 3) X,Y,Z
    coords_zyx  = field_vox[:, ::-1].T                     # (3, N) Z,Y,X

    displacements = np.column_stack([
        map_coordinates(disp_arr[..., i], coords_zyx, order=1, mode='nearest')
        for i in range(3)
    ])  # (N, 3) LPS displacements

    pts_lps = pts_lps + displacements

    # --- inv_affine (AffineTransform) ---
    inv_aff    = inv_composite.GetNthTransform(1)
    params     = np.array(inv_aff.GetParameters())
    M          = params[:9].reshape(3, 3)
    t          = params[9:]
    c          = np.array(inv_aff.GetFixedParameters()[:3])

    pts_lps = (pts_lps - c) @ M.T + c + t

    return pts_lps * np.array([-1., -1., 1.])  # LPS → RAS


# ---------------------------------------------------------------------------
# Amanpreet field inversion via sitk.InvertDisplacementField
# ---------------------------------------------------------------------------

def _sample_aps(affine_field, splines_field, sdiff_field, pts):
    """Apply composed forward map F(q) = affine[splines[sdiff[q]]] at pts (N,3) in Day0 voxel space."""
    def _s(field, p):
        coords = p.T
        fx = map_coordinates(field[2], coords, order=1, mode='nearest') - 1
        fy = map_coordinates(field[1], coords, order=1, mode='nearest') - 1
        fz = map_coordinates(field[0], coords, order=1, mode='nearest') - 1
        return np.stack([fx, fy, fz], axis=1)
    return _s(affine_field, _s(splines_field, _s(sdiff_field, pts)))


def _build_aps_inverse_sitk(day0_nib, day3_start_nib, affine_field, splines_field, sdiff_field):
    """
    Compute the Amanpreet inverse (Day3_start voxel → Day0 voxel) using
    sitk.InvertDisplacementField.

    Steps:
      1. Evaluate F on all Day0 voxels → Day3_start voxel positions
      2. Convert to a physical LPS displacement field on the Day0 grid
      3. Invert with sitk.InvertDisplacementField
      4. Sample the inverse at each Day3_start position → Day0 voxel coords

    Returns (Xs, Ys, Zs, 3) float32 array of Day0 0-indexed voxel coords,
    one entry per Day3_start voxel.
    """
    X, Y, Z = day0_nib.shape
    N = X * Y * Z

    # --- Forward field: Day0 voxel → Day3_start voxel ---
    print("  Computing composed Amanpreet forward field...")
    xi, yi, zi = np.mgrid[0:X, 0:Y, 0:Z]
    q = np.column_stack([xi.ravel(), yi.ravel(), zi.ravel()]).astype(np.float64)
    p_vox = _sample_aps(affine_field, splines_field, sdiff_field, q)  # (N, 3) Day3_start voxel

    # Convert both to LPS physical and compute displacement
    hom_q = np.column_stack([q,     np.ones(N)])
    hom_p = np.column_stack([p_vox, np.ones(N)])
    day0_lps     = ((day0_nib.affine     @ hom_q.T).T[:, :3] * np.array([-1., -1., 1.]))
    day3_lps     = ((day3_start_nib.affine @ hom_p.T).T[:, :3] * np.array([-1., -1., 1.]))
    disp_lps     = (day3_lps - day0_lps).astype(np.float64)  # (N, 3)

    # --- Build SimpleITK forward displacement field on Day0 grid ---
    disp_zyx = disp_lps.reshape(X, Y, Z, 3).transpose(2, 1, 0, 3)  # (Z, Y, X, 3)
    fwd_img  = sitk.GetImageFromArray(disp_zyx, isVector=True)

    spacing     = np.sqrt((day0_nib.affine[:3, :3]**2).sum(axis=0))
    ras_dir     = day0_nib.affine[:3, :3] / spacing
    lps_dir     = (ras_dir * np.array([[-1.], [-1.], [1.]])).flatten()
    origin_lps  = day0_nib.affine[:3, 3] * np.array([-1., -1., 1.])

    fwd_img.SetOrigin(origin_lps.tolist())
    fwd_img.SetSpacing(spacing.tolist())
    fwd_img.SetDirection(lps_dir.tolist())

    # --- Invert ---
    print("  Calling sitk.InvertDisplacementField...")
    inv_img  = sitk.InvertDisplacementField(fwd_img, maximumNumberOfIterations=50)
    inv_arr  = sitk.GetArrayFromImage(inv_img)  # (Z, Y, X, 3) LPS displacements

    # --- Sample inverse at every Day3_start voxel position ---
    Xs, Ys, Zs = day3_start_nib.shape
    Ns = Xs * Ys * Zs
    xis, yis, zis = np.mgrid[0:Xs, 0:Ys, 0:Zs]
    ps = np.column_stack([xis.ravel(), yis.ravel(), zis.ravel()]).astype(np.float64)

    # Day3_start voxel → physical LPS
    hom_s        = np.column_stack([ps, np.ones(Ns)])
    day3_lps_s   = ((day3_start_nib.affine @ hom_s.T).T[:, :3] * np.array([-1., -1., 1.]))

    # Convert Day3_start LPS to inv_field voxel coords (Z,Y,X ordering)
    inv_origin    = np.array(inv_img.GetOrigin())
    inv_spacing   = np.array(inv_img.GetSpacing())
    inv_direction = np.array(inv_img.GetDirection()).reshape(3, 3)
    p2v           = np.linalg.inv(inv_direction @ np.diag(inv_spacing))
    inv_vox_xyz   = (day3_lps_s - inv_origin) @ p2v.T   # (Ns, 3) X,Y,Z
    inv_coords    = inv_vox_xyz[:, ::-1].T               # (3, Ns) Z,Y,X

    inv_disp = np.column_stack([
        map_coordinates(inv_arr[..., i], inv_coords, order=1, mode='nearest')
        for i in range(3)
    ])  # (Ns, 3) LPS displacement

    # Day0 physical = Day3_start physical + inverse displacement
    day0_lps_s   = day3_lps_s + inv_disp
    day0_ras_s   = day0_lps_s * np.array([-1., -1., 1.])
    hom_d        = np.column_stack([day0_ras_s, np.ones(Ns)])
    day0_vox_out = (np.linalg.inv(day0_nib.affine) @ hom_d.T).T[:, :3].astype(np.float32)

    return day0_vox_out.reshape(Xs, Ys, Zs, 3)


# ---------------------------------------------------------------------------
# Core: build dense Day3_end → Day0 displacement field
# ---------------------------------------------------------------------------

def build_day3end_to_day0_field(day3_end_nib, day3_start_nib, inv_composite,
                                 affine_field, splines_field, sdiff_field,
                                 n_iter=N_ITER, alpha=ALPHA):
    """
    For every voxel in Day3_end space compute the corresponding 0-indexed Day0
    voxel coordinate via:
        Day3_end voxel → physical RAS
        → [Slicer inv] → Day3_start physical RAS
        → Day3_start voxel
        → [F^{-1}]    → Day0 voxel

    Returns (X, Y, Z, 3) float32 array of Day0 voxel coords.
    """
    X, Y, Z = day3_end_nib.shape
    N = X * Y * Z
    print(f"  Grid size: {X}×{Y}×{Z} = {N:,} voxels")

    # Dense grid of Day3_end voxel indices
    xi, yi, zi = np.mgrid[0:X, 0:Y, 0:Z]
    vox = np.column_stack([xi.ravel(), yi.ravel(), zi.ravel()]).astype(np.float32)

    # Day3_end voxel → physical RAS
    hom      = np.column_stack([vox, np.ones(N, dtype=np.float32)])
    phys_ras = (day3_end_nib.affine @ hom.T).T[:, :3].astype(np.float32)
    del hom

    # Physical RAS → Day3_start physical RAS (Slicer inverse, vectorized)
    print("  Applying Slicer inverse transform...")
    day3start_ras = _apply_slicer_inv_vectorized(phys_ras, inv_composite).astype(np.float32)
    del phys_ras

    # Physical RAS → Day3_start voxel (0-indexed)
    hom            = np.column_stack([day3start_ras, np.ones(N, dtype=np.float32)])
    day3start_vox  = (np.linalg.inv(day3_start_nib.affine) @ hom.T).T[:, :3].astype(np.float32)
    del day3start_ras, hom

    if APS_INV_METHOD == 'sitk':
        # Build inverse on Day3_start grid via sitk, then sample at Slicer-computed positions
        day0_nib = nib.load(sorted(glob.glob(os.path.join(DAY0_DIR, '*.nii.gz')))[0])
        aps_inv  = _build_aps_inverse_sitk(day0_nib, day3_start_nib,
                                            affine_field, splines_field, sdiff_field)
        # aps_inv: (Xs, Ys, Zs, 3) Day0 voxel coords for each Day3_start voxel
        # Sample it at the Slicer-computed Day3_start positions
        print("  Sampling Amanpreet inverse at Slicer-computed Day3_start positions...")
        coords_s = day3start_vox[:, ::-1].T  # (3, N) Z,Y,X for map_coordinates
        day0_vox = np.column_stack([
            map_coordinates(aps_inv[..., i], coords_s, order=1, mode='constant', cval=-1)
            for i in range(3)
        ]).astype(np.float32)

    else:  # fixedpoint
        field_bounds = np.array([383., 143., 511.])
        in_bounds = np.all((day3start_vox >= 0) & (day3start_vox <= field_bounds), axis=1)
        n_in = in_bounds.sum()
        print(f"  {n_in:,} / {N:,} voxels in Day0 field bounds — masking the rest to -1")

        day0_vox = np.full((N, 3), -1., dtype=np.float32)
        if n_in > 0:
            print(f"  Inverting Amanpreet fields ({n_iter} iterations, alpha={alpha})...")
            print(day3start_vox.shape)
            day0_vox[in_bounds] = propagate_tiles_to_day0(
                day3start_vox[in_bounds].reshape(1, n_in, 3).astype(np.float64),
                affine_field, splines_field, sdiff_field,
                n_iter=n_iter, alpha=alpha, verbose=True
            ).reshape(n_in, 3).astype(np.float32)

    return day0_vox.reshape(X, Y, Z, 3)


# ---------------------------------------------------------------------------
# Core: resample a volume using a pre-computed voxel coordinate field
# ---------------------------------------------------------------------------

def resample_with_field(vol_arr, voxel_field, order=1):
    """
    Sample vol_arr (X, Y, Z) at the positions given by voxel_field (X, Y, Z, 3).
    Positions outside the source volume are filled with zero.
    Returns a resampled array with the same shape as voxel_field[:3].
    """
    coords = voxel_field.reshape(-1, 3).T  # (3, N)
    return map_coordinates(vol_arr, coords, order=order, mode='constant', cval=0).reshape(voxel_field.shape[:3])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    day3_end_nib   = nib.load(DAY3_END_PATH)
    day3_start_nib = nib.load(DAY3_START_STRUCT)
    inv_composite  = _load_transform(SLICER_INV_CACHE)

    # --- Day3_start extras → Day3_end (Slicer only) ---
    for path in DAY3_START_EXTRA:
        if not os.path.exists(path):
            print(f"Skipping (not found): {path}")
            continue
        stem = os.path.basename(path).replace('.nii.gz', '')
        out  = os.path.join(OUT_DIR, f'{stem}_regToDay3End.nii.gz')
        print(f"Resampling {stem} → Day3_end ...")
        _interp = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear}[INTERP_ORDER]
        arr = ApplySlicerTransform(path, DAY3_END_PATH, SLICER_INV_CACHE, interpolator=_interp)
        nib.save(nib.Nifti1Image(arr, day3_end_nib.affine, day3_end_nib.header), out)
        print(f"  Saved → {out}")

    # --- Build / load Day3_end → Day0 displacement field ---
    if os.path.exists(DISPLACEMENT_CACHE):
        print(f"Loading cached displacement field from {DISPLACEMENT_CACHE} ...")
        voxel_field = np.load(DISPLACEMENT_CACHE)
    else:
        affine_field  = np.load(AFFINE_FIELD_PATH)
        splines_field = np.load(SPLINES_FIELD_PATH)
        sdiff_field   = np.load(SDIFF_FIELD_PATH)

        print("Building Day3_end → Day0 displacement field (one-time cost) ...")
        voxel_field = build_day3end_to_day0_field(
            day3_end_nib, day3_start_nib, inv_composite,
            affine_field, splines_field, sdiff_field,
            n_iter=10
        )
        np.save(DISPLACEMENT_CACHE, voxel_field)
        print(f"  Cached → {DISPLACEMENT_CACHE}")

    # --- Day0 volumes → Day3_end ---
    day0_paths = sorted(glob.glob(os.path.join(DAY0_DIR, '*.nii.gz')))
    for path in day0_paths:
        stem = os.path.basename(path).replace('.nii.gz', '')
        out  = os.path.join(OUT_DIR, f'{stem}_regToDay3End.nii.gz')
        print(f"Resampling Day0/{stem} → Day3_end ...")
        vol_arr = nib.load(path).get_fdata(dtype=np.float32)
        arr     = resample_with_field(vol_arr, voxel_field, order=INTERP_ORDER)
        nib.save(nib.Nifti1Image(arr, day3_end_nib.affine, day3_end_nib.header), out)
        print(f"  Saved → {out}")

    print("All done.")


# ---------------------------------------------------------------------------
# Ad-hoc: resample any single volume to Day3_end
# ---------------------------------------------------------------------------
#
# def resample_single(vol_path, timepoint, out_dir=None, order=1):
#     """
#     Resample an arbitrary volume to Day3_end space.
#
#     timepoint=3  →  Day3_start volume: apply Slicer inverse only
#     timepoint=0  →  Day0 volume:       apply Slicer inverse + Amanpreet F^{-1}
#
#     out_dir: directory to save the result. Defaults to the same folder as vol_path.
#              Output filename is the original name with '_rd3e' appended.
#     order: interpolation order (1=linear, 0=nearest neighbour)
#     """
#     stem     = os.path.basename(vol_path).replace('.nii.gz', '').replace('.nii', '')
#     out_dir  = out_dir or os.path.dirname(vol_path)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f'{stem}_rd3e.nii.gz')
#
#     day3_end_nib  = nib.load(DAY3_END_PATH)
#     inv_composite = _load_transform(SLICER_INV_CACHE)
#
#     _sitk_interp = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear}
#     if timepoint == 3:
#         print(f"Resampling (Day3_start → Day3_end): {os.path.basename(vol_path)}")
#         arr = ApplySlicerTransform(vol_path, DAY3_END_PATH, SLICER_INV_CACHE, interpolator=_sitk_interp[order])
#
#     elif timepoint == 0:
#         print(f"Resampling (Day0 → Day3_end): {os.path.basename(vol_path)}")
#
#         if os.path.exists(DISPLACEMENT_CACHE):
#             voxel_field = np.load(DISPLACEMENT_CACHE)
#         else:
#             day3_start_nib = nib.load(DAY3_START_STRUCT)
#             affine_field   = np.load(AFFINE_FIELD_PATH)
#             splines_field  = np.load(SPLINES_FIELD_PATH)
#             sdiff_field    = np.load(SDIFF_FIELD_PATH)
#             print("  Building displacement field (one-time)...")
#             voxel_field = build_day3end_to_day0_field(
#                 day3_end_nib, day3_start_nib, inv_composite,
#                 affine_field, splines_field, sdiff_field, n_iter=N_ITER, alpha=ALPHA
#             )
#             os.makedirs(OUT_DIR, exist_ok=True)
#             np.save(DISPLACEMENT_CACHE, voxel_field)
#
#         vol_arr = nib.load(vol_path).get_fdata(dtype=np.float32)
#         arr     = resample_with_field(vol_arr, voxel_field, order=order)
#
#     else:
#         raise ValueError(f"timepoint must be 0 (Day0) or 3 (Day3_start), got {timepoint}")
#
#     nib.save(nib.Nifti1Image(arr, day3_end_nib.affine, day3_end_nib.header), out_path)
#     print(f"  Saved → {out_path}")
#
#
# # Example ad-hoc calls — uncomment and edit paths as needed:
# #Change order to 1 for linear interp, 0 for NN
# resample_single("/Users/jbonaventura/Desktop/Annotations/Day3Transformed/Def/dsrHnE_R23-055_H7_7a_5X_InVivo_tiles2_reg2start_resampled.nii.gz", 0, "/Users/jbonaventura/Desktop/Annotations/out", order=INTERP_ORDER)
