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
from nibabel.processing import resample_from_to
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
import csv
import cv2
import matplotlib.pyplot as plt
from ApplyTransforms import ApplySlicerTransform, propagate_tiles_to_day0
from RabbitPathFinder import find_nii

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RABBIT_FOLDER     = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
READY_RABBITS_CSV = os.path.join(os.path.dirname(RABBIT_FOLDER), 'ReadyRabbits.csv')

# Per-rabbit paths (_base, DAY3_DIR, DAY0_DIR, XFMS_DIR, OUT_DIR, and everything
# derived from them below) are set as globals inside run_for_rabbit() so the
# batch loop at the bottom of the file can re-point them at each rabbit in turn.

# ---------------------------------------------------------------------------
# Load Day3 filenames from log CSV
# ---------------------------------------------------------------------------
def _load_day3_log(day3_dir, rabbit_id):
    log_name = rabbit_id.replace('-', '_') + '_InVDay3Log.csv'
    log_path = os.path.join(day3_dir, log_name)
    roles = {}
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            key = row['Home'].strip()
            val = os.path.join(day3_dir, row['File Name'].strip() + '.nii.gz')
            if key == 'Skip':
                print(f"  Skipping (flagged): {os.path.basename(val)}")
            roles.setdefault(key, []).append(val)
    return roles


def _load_rabbit_ids(csv_path):
    """Load the batch rabbit-ID list from ReadyRabbits.csv ('rabbit' column)."""
    with open(csv_path, newline='') as f:
        return [row['rabbit'].strip() for row in csv.DictReader(f)]

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
# Masking config
# ---------------------------------------------------------------------------
# Set APPLY_MASK=False to skip masking entirely and save raw registered volumes.
# Set RETUNE_MASK=True to interactively tune parameters before each run.
# Set RETUNE_MASK=False to use the values below directly (no prompts).
APPLY_MASK      = True
RETUNE_MASK     = False
MASK_THRESH     = 10
MASK_MASKTHRESH = 150
MASK_KERNSIZE   = 15

# Set RUN_DAY0=False to skip the Day3_end -> Day0 displacement field and Day0
# resampling entirely -- only the Day3_end structural + Day3_start extras
# (T1/T2 maps etc.) get built.
RUN_DAY0        = False

# T1/T2 map coronal-slice artifact masking (applied to the native volume,
# before any resampling/transform). Some acquisitions have one or more whole
# corrupted coronal slices (extreme bright or negative garbage values across
# nearly the entire slice) -- confirmed present on multiple rabbits at
# different slice indices each time, unlike a fixed-position issue. Set
# MASK_TMAP_ARTIFACTS=False to skip this and resample T1/T2 raw.
MASK_TMAP_ARTIFACTS = True
TMAP_Z_THRESH       = 15  # robust (median/MAD) |z|-score cutoff on a coronal
                          # slice's median intensity -- real tissue variation
                          # seen so far tops out around |z|~12, confirmed
                          # artifacts run |z|~50-18000, so 15 cleanly separates
                          # the two with margin on both sides.

# T1 maps specifically (T2 maps look fine) also have widespread, milder
# slab-edge artifacts -- degraded fits on the first/last slice of the
# tissue-containing range -- too mild and inconsistent (sometimes a
# near-zero clamp, sometimes an elevated-but-still-varying plane) to
# reliably clear TMAP_Z_THRESH. Rather than chase that with a lower
# threshold, unconditionally zero the outermost tissue-bearing slice on
# each end for T1 maps, after the coronal-slice masking above.

# ---------------------------------------------------------------------------
# Normalization config
# ---------------------------------------------------------------------------
# Set SAVE_NORMALIZED=True to additionally save an intensity-normalized copy of
# each registered volume into a Normalized/ subfolder of OUT_DIR - the raw save
# next to it is untouched either way.
SAVE_NORMALIZED = True
NORM_METHOD     = 'percentile'  # 'percentile' | 'minmax' | 'zscore'
NORM_CLIP_PCT   = 1.0
NORM_OUT_RANGE  = (0.0, 1.0)

# Set SAVE_NORMALIZED_METHOD2=True to also save a copy into Normalized_method2/:
#   T1/T2 maps     -> fixed physical scale (ms / QUANT_MAP_SCALE), clipped to [0, 1]
#   other contrasts -> ratio to the rabbit's healthy-muscle median, so healthy
#                      muscle = 1 and background stays 0 (downstream treats 0 as
#                      outside body/FOV). Dividing removes per-rabbit receiver
#                      gain / coil-sensitivity scaling.
# Healthy-muscle mask: hand-drawn in Slicer, one per rabbit, at
# InVMRDataSets/HEALTHY_MUSCLE_MASK_NAME (any grid; resampled by affine).
SAVE_NORMALIZED_METHOD2  = True
QUANT_MAP_SCALE          = {'t1map': 3000.0, 't2map': 200.0}
HEALTHY_MUSCLE_MASK_NAME = 'Healthy_Muscle_Annotation.nii.gz'


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


def _apply_slicer_transform_vectorized(pts_ras, slicer_transform):
    """
    Apply the Slicer Day3_start_to_end.h5 transform to an (N, 3) array of
    physical RAS points, fully vectorized via map_coordinates. Per ITK's
    fixed->moving pull convention (registered Day3_start (moving) -> Day3_end
    (fixed)), applying this transform to a Day3_end point pulls the
    corresponding Day3_start point -- no separate inversion step needed.

    The transform applies its displacement field first, then its affine (both in LPS).
    Returns (N, 3) physical RAS points in Day3_start space.
    """
    pts_lps = pts_ras * np.array([-1., -1., 1.])

    # --- displacement field (DisplacementFieldTransform) ---
    disp_t     = sitk.DisplacementFieldTransform(slicer_transform.GetNthTransform(0))
    disp_img   = disp_t.GetDisplacementField()
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

    # --- affine (AffineTransform) ---
    aff        = slicer_transform.GetNthTransform(1)
    params     = np.array(aff.GetParameters())
    M          = params[:9].reshape(3, 3)
    t          = params[9:]
    c          = np.array(aff.GetFixedParameters()[:3])

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
# Grid resampling (no transform)
# ---------------------------------------------------------------------------

def resample_to_grid(path, reference_sitk, interpolator=sitk.sitkLinear):
    """Resample a volume onto the reference grid using an identity transform."""
    moving = sitk.ReadImage(path)
    resampled = sitk.Resample(moving, reference_sitk, sitk.Transform(),
                              interpolator, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def build_body_mask(vol_arr, thresh, maskthresh, kernsize):
    if np.max(vol_arr) == 0:
        return np.zeros(vol_arr.shape, dtype=bool)
    pic = np.array(vol_arr / np.max(vol_arr) * 255).astype(np.uint8)
    thresh_pic = np.where(pic > thresh, 1000, pic).astype(np.float32)
    kernel = np.ones((kernsize, kernsize), np.float32) / (kernsize * kernsize)
    filtered = np.zeros(thresh_pic.shape, dtype=np.float32)
    for k in range(thresh_pic.shape[2]):
        filtered[:, :, k] = cv2.filter2D(thresh_pic[:, :, k], ddepth=-1, kernel=kernel)
    return filtered > maskthresh


def tune_mask_interactively(vol_arr, thresh, maskthresh, kernsize):
    while True:
        mask = build_body_mask(vol_arr, thresh, maskthresh, kernsize)
        mid = [s // 2 for s in vol_arr.shape]
        pic = np.array(vol_arr / np.max(vol_arr) * 255).astype(np.uint8)
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for col, (sl_img, sl_mask) in enumerate([
            (pic[mid[0], :, :], mask[mid[0], :, :]),
            (pic[:, mid[1], :], mask[:, mid[1], :]),
            (pic[:, :, mid[2]], mask[:, :, mid[2]]),
        ]):
            axs[0, col].imshow(sl_img, cmap='gray')
            axs[1, col].imshow(sl_img * sl_mask, cmap='gray')
        plt.suptitle(f'thresh={thresh}  maskthresh={maskthresh}  kernsize={kernsize}'
                     f'\nTop: original    Bottom: masked')
        plt.tight_layout()
        plt.show()
        if input('Accept mask? (y/n): ').strip().lower() == 'y':
            return mask, thresh, maskthresh, kernsize
        val = input(f'  thresh [{thresh}]: ').strip()
        thresh = float(val) if val else thresh
        val = input(f'  maskthresh [{maskthresh}]: ').strip()
        maskthresh = float(val) if val else maskthresh
        val = input(f'  kernsize [{kernsize}]: ').strip()
        kernsize = int(val) if val else kernsize


def is_t_map(path):
    stem = os.path.basename(path).lower()
    return 't1map' in stem or 't2map' in stem


def flag_bad_coronal_slices(arr, z_thresh=TMAP_Z_THRESH):
    """Two-sided robust-outlier test on each coronal slice's (axis 2) median
    intensity, using only finite nonzero voxels -- flags whole corrupted
    slices (either abnormally bright or abnormally dark/negative), not just
    background. Returns (bad, z): bad is a (Z,) bool array, z the per-slice
    robust z-score (median/MAD) for inspection."""
    finite_nonzero = np.isfinite(arr) & (arr != 0)
    n = arr.shape[2]
    slice_med = np.full(n, np.nan)
    for k in range(n):
        vals = arr[:, :, k][finite_nonzero[:, :, k]]
        if vals.size > 20:
            slice_med[k] = np.median(vals)

    valid = np.isfinite(slice_med)
    med = np.median(slice_med[valid])
    mad = np.median(np.abs(slice_med[valid] - med)) + 1e-9
    z = np.full(n, np.nan)
    z[valid] = (slice_med[valid] - med) / (1.4826 * mad)

    bad = np.zeros(n, dtype=bool)
    bad[valid] = np.abs(z[valid]) > z_thresh
    return bad, z


def mask_bad_coronal_slices(path, z_thresh=TMAP_Z_THRESH, out_dir=None):
    """Loads a T1/T2 map, zeros out any coronal slice flagged as a corrupted
    outlier plane, and saves a copy for downstream resampling to use in
    place of the raw file. Returns the new path, or the original path
    unchanged if nothing was flagged."""
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    bad, z = flag_bad_coronal_slices(arr, z_thresh)
    if not bad.any():
        return path

    flagged = np.where(bad)[0]
    print(f"  [{os.path.basename(path)}] masking {bad.sum()} corrupted coronal "
          f"slice(s): {list(flagged)} (z={[f'{z[i]:.0f}' for i in flagged]})")
    arr[:, :, bad] = 0.0

    stem = os.path.basename(path).replace('.nii.gz', '')
    out_dir = out_dir or tempfile.gettempdir()
    out_path = os.path.join(out_dir, f'{stem}_slicemasked.nii.gz')
    nib.save(nib.Nifti1Image(arr, img.affine, img.header), out_path)
    return out_path


def is_t1_map(path):
    return 't1map' in os.path.basename(path).lower()


def zero_edge_tissue_planes(path, out_dir=None):
    """Zero the first and last coronal (axis 2) slice containing any nonzero
    voxels. T1 map slab-edge slices have degraded fits too mild/inconsistent
    to reliably clear flag_bad_coronal_slices' outlier threshold, so these
    are stripped unconditionally instead of via the z-score test."""
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    tissue_slices = np.where(np.any(arr != 0, axis=(0, 1)))[0]
    if tissue_slices.size < 3:
        return path

    first, last = tissue_slices[0], tissue_slices[-1]
    print(f"  [{os.path.basename(path)}] zeroing edge coronal slices {first} and {last}")
    arr[:, :, first] = 0.0
    arr[:, :, last] = 0.0

    stem = os.path.basename(path).replace('.nii.gz', '')
    out_dir = out_dir or tempfile.gettempdir()
    out_path = os.path.join(out_dir, f'{stem}_edgezeroed.nii.gz')
    nib.save(nib.Nifti1Image(arr, img.affine, img.header), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Core: build dense Day3_end → Day0 displacement field
# ---------------------------------------------------------------------------

def build_day3end_to_day0_field(day3_end_nib, day3_start_nib, slicer_transform,
                                 affine_field, splines_field, sdiff_field,
                                 n_iter=N_ITER, alpha=ALPHA):
    """
    For every voxel in Day3_end space compute the corresponding 0-indexed Day0
    voxel coordinate via:
        Day3_end voxel → physical RAS
        → [Slicer transform] → Day3_start physical RAS
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

    # Physical RAS → Day3_start physical RAS (Slicer transform, vectorized)
    print("  Applying Slicer transform...")
    day3start_ras = _apply_slicer_transform_vectorized(phys_ras, slicer_transform).astype(np.float32)
    del phys_ras

    # Physical RAS → APS voxel (0-indexed).
    # The APS fields' domain is whatever raw on-disk orientation the Day0
    # volumes are stored in (loaded with no reorientation below) — NOT
    # nibabel's canonical RAS+. Reorient Day3_start to match Day0's native
    # orientation (rather than forcing RAS+) to get an affine usable for this
    # lookup, since Day3_start/Day3_end on-disk orientation varies per rabbit
    # and can't be trusted directly. Verified to reproduce the known-good
    # affine exactly on rabbits where a native APS-oriented Day3 volume also
    # exists for comparison.
    day0_ref_nib = nib.load(sorted(glob.glob(os.path.join(DAY0_DIR, 'Masked', '*.nii.gz')))[0])
    ornt_xfm     = nib.orientations.ornt_transform(
        nib.io_orientation(day3_end_nib.affine),
        nib.io_orientation(day0_ref_nib.affine),
    )
    aps_affine     = day3_end_nib.as_reoriented(ornt_xfm).affine
    hom            = np.column_stack([day3start_ras, np.ones(N, dtype=np.float32)])
    day3start_vox  = (np.linalg.inv(aps_affine) @ hom.T).T[:, :3].astype(np.float32)
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
        print(f"  Inverting Amanpreet fields ({n_iter} iterations, alpha={alpha})...")
        day0_vox = propagate_tiles_to_day0(
            day3start_vox.reshape(1, N, 3).astype(np.float64),
            affine_field, splines_field, sdiff_field,
            n_iter=n_iter, alpha=alpha, verbose=True
        ).reshape(N, 3).astype(np.float32)

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
# Normalization (adapted from mri_embed.py's _load_norm, minus the npy-file load)
# ---------------------------------------------------------------------------

def normalize_mr_array(arr, norm='percentile', clip_pct=1.0, out_range=(0.0, 1.0)):
    """Intensity-normalize an already-loaded MR array: percentile clip, minmax, or zscore."""
    a = np.nan_to_num(arr.astype(np.float32))
    lo_out, hi_out = out_range
    nz = a[a != 0]
    if norm in ('percentile', 'clip_minmax'):
        if nz.size:
            lo, hi = np.percentile(nz, [clip_pct, 100 - clip_pct])
            a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * (hi_out - lo_out) + lo_out
    elif norm == 'minmax':
        if nz.size:
            lo, hi = float(nz.min()), float(nz.max())
            a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * (hi_out - lo_out) + lo_out
    elif norm == 'zscore':
        a = (a - a.mean()) / (a.std() + 1e-9)
    return a


def load_healthy_muscle_mask(mask_path, day3_end_nib):
    """Load the healthy-muscle label volume as a bool mask on the Day3_end grid.
    Masks exported on a cropped grid are resampled (nearest neighbour, by affine)."""
    mask_nib = nib.load(mask_path)
    if mask_nib.shape != day3_end_nib.shape or not np.allclose(mask_nib.affine, day3_end_nib.affine):
        mask_nib = resample_from_to(mask_nib, day3_end_nib, order=0)
    mask = np.asarray(mask_nib.dataobj) > 0
    print(f'  Healthy-muscle mask: {mask.sum():,} voxels ({mask_path})')
    return mask


def normalize_method2(arr, stem, muscle_mask):
    """T1/T2 maps: divide by a fixed physical scale and clip to [0, 1].
    Other contrasts: divide by the healthy-muscle median (nonzero voxels only),
    so healthy muscle = 1 and zero background stays zero."""
    a = np.nan_to_num(arr.astype(np.float32))
    stem_lower = stem.lower()
    for key, scale in QUANT_MAP_SCALE.items():
        if key in stem_lower:
            return np.clip(a / scale, 0.0, 1.0)

    ref = a[muscle_mask & (a != 0)]
    if ref.size == 0:
        raise ValueError(f'{stem}: no nonzero voxels inside the healthy-muscle mask')
    median = float(np.median(ref))
    print(f'  [{stem}] healthy-muscle median = {median:.1f} (n={ref.size:,})')
    return a / median


def save_registered(arr, stem, day3_end_nib, out_dir, norm_out_dir=None,
                    norm2_out_dir=None, muscle_mask=None):
    """Save the raw registered array, plus optional normalized copies in
    norm_out_dir (per-volume percentile) and norm2_out_dir (method 2)."""
    out = os.path.join(out_dir, f'{stem}_regToDay3End.nii.gz')
    nib.save(nib.Nifti1Image(arr.astype(np.float32), day3_end_nib.affine), out)
    print(f"  Saved → {out}")
    if norm_out_dir:
        norm_arr = normalize_mr_array(arr, NORM_METHOD, NORM_CLIP_PCT, NORM_OUT_RANGE)
        norm_out = os.path.join(norm_out_dir, f'{stem}_regToDay3End.nii.gz')
        nib.save(nib.Nifti1Image(norm_arr.astype(np.float32), day3_end_nib.affine), norm_out)
        print(f"  Saved normalized → {norm_out}")
    if norm2_out_dir:
        norm2_arr = normalize_method2(arr, stem, muscle_mask)
        norm2_out = os.path.join(norm2_out_dir, f'{stem}_regToDay3End.nii.gz')
        nib.save(nib.Nifti1Image(norm2_arr.astype(np.float32), day3_end_nib.affine), norm2_out)
        print(f"  Saved normalized (method 2) → {norm2_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_rabbit(rabbit_id):
    global _base, DAY3_DIR, DAY0_DIR, XFMS_DIR, OUT_DIR
    global SLICER_TRANSFORM_PATH
    global AFFINE_FIELD_PATH, SPLINES_FIELD_PATH, SDIFF_FIELD_PATH, DISPLACEMENT_CACHE
    global MASK_THRESH, MASK_MASKTHRESH, MASK_KERNSIZE

    _base     = os.path.join(RABBIT_FOLDER, rabbit_id, 'InVivo_MR', 'InVMRDataSets')
    DAY3_DIR  = os.path.join(_base, 'Day3')
    DAY0_DIR  = os.path.join(_base, 'Day0')
    XFMS_DIR  = os.path.join(_base, 'Transforms')
    OUT_DIR   = os.path.join(_base, 'Day3End_Registered')

    # Registered directly as Day3_start (moving) -> Day3_end (fixed) in Slicer, so
    # in ITK's fixed->moving pull convention this is already the transform
    # ApplySlicerTransform needs -- no algebraic/displacement-field inversion step.
    SLICER_TRANSFORM_PATH = os.path.join(DAY3_DIR, 'Day3_start_to_end.h5')
    AFFINE_FIELD_PATH  = os.path.join(XFMS_DIR, 'Affine_deformation.npy')
    SPLINES_FIELD_PATH = os.path.join(XFMS_DIR, 'SplinesProjection.npy')
    SDIFF_FIELD_PATH   = os.path.join(XFMS_DIR, 'sdiff.npy')
    DISPLACEMENT_CACHE = os.path.join(OUT_DIR, 'day3end_to_day0_voxel_coords.npy')

    print(f"\n=== {rabbit_id} ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    NORM_OUT_DIR = os.path.join(OUT_DIR, 'Normalized')
    if SAVE_NORMALIZED:
        os.makedirs(NORM_OUT_DIR, exist_ok=True)
    NORM2_OUT_DIR = os.path.join(OUT_DIR, 'Normalized_method2')
    if SAVE_NORMALIZED_METHOD2:
        os.makedirs(NORM2_OUT_DIR, exist_ok=True)

    _day3 = _load_day3_log(DAY3_DIR, rabbit_id)
    DAY3_END_PATH     = _day3['End'][0]
    DAY3_START_STRUCT = _day3['Fixed'][0]
    DAY3_START_EXTRA  = _day3.get('Start', [])
    print("Day3 Extras- ", DAY3_START_EXTRA)

    # Grid/affine for every output comes from the top-level InVivo_MR/*.nii(.gz) file,
    # not the Day3 log's own 'End' copy (DAY3_END_PATH, kept below only for its stem —
    # used to name the saved structural output). Both files hold the identical scan
    # (confirmed pixel-identical after axis permutation) but were exported on disk with
    # different array axis conventions; MakeVoxelVolV2.py and ApplyTransforms.py's tile
    # propagation already treat the top-level file as the canonical 'InVivo' reference,
    # so anchoring here on the same file keeps every downstream grid consistent instead
    # of introducing a second, conflicting convention.
    INVIVO_REF_PATH = str(find_nii(os.path.join(RABBIT_FOLDER, rabbit_id, 'InVivo_MR')))
    day3_end_nib   = nib.load(INVIVO_REF_PATH)
    print(f"Day3End reference grid: {INVIVO_REF_PATH}  (naming stem from: {DAY3_END_PATH})")
    day3_start_nib = nib.load(DAY3_START_STRUCT)
    if not os.path.exists(SLICER_TRANSFORM_PATH):
        raise FileNotFoundError(f"{SLICER_TRANSFORM_PATH} not found")
    print(f'Loading {os.path.basename(SLICER_TRANSFORM_PATH)}...')
    slicer_transform = _load_transform(SLICER_TRANSFORM_PATH)

    # Normalized-copy settings shared by every save_registered() call below
    norm_kwargs = {'norm_out_dir': NORM_OUT_DIR if SAVE_NORMALIZED else None}
    if SAVE_NORMALIZED_METHOD2:
        norm_kwargs['norm2_out_dir'] = NORM2_OUT_DIR
        norm_kwargs['muscle_mask'] = load_healthy_muscle_mask(
            os.path.join(_base, HEALTHY_MUSCLE_MASK_NAME), day3_end_nib)

    # --- Build body mask from Day3_end structural ---
    if APPLY_MASK:
        day3_end_arr = day3_end_nib.get_fdata(dtype=np.float32)
        if RETUNE_MASK:
            mask, MASK_THRESH, MASK_MASKTHRESH, MASK_KERNSIZE = tune_mask_interactively(
                day3_end_arr, MASK_THRESH, MASK_MASKTHRESH, MASK_KERNSIZE
            )
        else:
            mask = build_body_mask(day3_end_arr, MASK_THRESH, MASK_MASKTHRESH, MASK_KERNSIZE)
        print(f'  Body mask built: {mask.sum():,} voxels retained')
    else:
        day3_end_arr = day3_end_nib.get_fdata(dtype=np.float32)

    # --- Day3_end structural itself → Day3End_Registered (already on the Day3_end
    # grid, no resampling needed, but still needs masking/saving/normalizing like
    # every other volume so it's available for cross-contrast comparison) ---
    day3_end_stem = os.path.basename(DAY3_END_PATH).replace('.nii.gz', '')
    end_arr = day3_end_arr * mask if APPLY_MASK else day3_end_arr
    save_registered(end_arr, day3_end_stem, day3_end_nib, OUT_DIR, **norm_kwargs)
    del day3_end_arr, end_arr

    # --- Day3_start extras → Day3_end (Slicer only) ---
    for path in DAY3_START_EXTRA:
        if not os.path.exists(path):
            print(f"Skipping (not found): {path}")
            continue
        stem = os.path.basename(path).replace('.nii.gz', '')
        print(path)
        print(nib.load(path).header.get_data_dtype())
        h = nib.load(path).header
        print(h.get_slope_inter())  # (scl_slope, scl_inter)
        resample_path = path
        if MASK_TMAP_ARTIFACTS and is_t_map(path):
            resample_path = mask_bad_coronal_slices(path)
            if is_t1_map(path):
                resample_path = zero_edge_tissue_planes(resample_path)
        print(f"Resampling {stem} → Day3_end ...")
        _interp = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear}[INTERP_ORDER]
        arr = ApplySlicerTransform(resample_path, INVIVO_REF_PATH, SLICER_TRANSFORM_PATH, interpolator=_interp)

        print(f"  arr dtype: {arr.dtype}")
        print(f"  unique values sample: {np.unique(arr[arr > 0])[:20]}")
        print(f"  min step: {np.diff(np.unique(arr[arr > 0])).min():.6f}")

        if APPLY_MASK:
            arr = arr * mask
        save_registered(arr, stem, day3_end_nib, OUT_DIR, **norm_kwargs)

    # --- End_native volumes → Day3_end grid (identity resample only) ---
    day3_end_sitk = sitk.ReadImage(INVIVO_REF_PATH)
    _interp = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear}[INTERP_ORDER]
    for path in _day3.get('End_native', []):
        if not os.path.exists(path):
            print(f"Skipping (not found): {path}")
            continue
        stem = os.path.basename(path).replace('.nii.gz', '')
        resample_path = path
        if MASK_TMAP_ARTIFACTS and is_t_map(path):
            resample_path = mask_bad_coronal_slices(path)
            if is_t1_map(path):
                resample_path = zero_edge_tissue_planes(resample_path)
        print(f"Resampling {stem} → Day3_end grid (no transform) ...")
        arr = resample_to_grid(resample_path, day3_end_sitk, interpolator=_interp)
        if APPLY_MASK:
            arr = arr * mask
        save_registered(arr, stem, day3_end_nib, OUT_DIR, **norm_kwargs)

    if RUN_DAY0:
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
                day3_end_nib, day3_start_nib, slicer_transform,
                affine_field, splines_field, sdiff_field
            )
            np.save(DISPLACEMENT_CACHE, voxel_field)
            print(f"  Cached → {DISPLACEMENT_CACHE}")

        # --- Day0 volumes → Day3_end ---
        day0_paths = sorted(glob.glob(os.path.join(DAY0_DIR, 'Masked', '*.nii.gz')))
        for path in day0_paths:
            stem = os.path.basename(path).replace('.nii.gz', '')
            print(f"Resampling Day0/{stem} → Day3_end ...")
            vol_arr = nib.load(path).get_fdata(dtype=np.float32)
            arr     = resample_with_field(vol_arr, voxel_field, order=INTERP_ORDER)
            if APPLY_MASK:
                arr = arr * mask
            save_registered(arr, stem, day3_end_nib, OUT_DIR, **norm_kwargs)
    else:
        print("RUN_DAY0=False -- skipping Day0 displacement field and resampling.")

    print(f"All done ({rabbit_id}).")


if __name__ == '__main__':
    for _rabbit_id in _load_rabbit_ids(READY_RABBITS_CSV):
        run_for_rabbit(_rabbit_id)


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
