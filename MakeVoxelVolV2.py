import os
import tempfile
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from RabbitPathFinder import find_all_the_paths
from FinalRegProcessingPipeline import MultiStepReg
from ComposedTransformReg import ComposedReg
from ApplyTransforms import nib_to_sitk
from NiftyTransforms import makesegfromvol

# ---------------------------------------------------------------------------
# Config — edit these per run
# ---------------------------------------------------------------------------
RABBIT_FOLDER = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RABBIT_ID     = 'R24-058'
BLOCK         = 6
END_FIXED     = 'BlockFace'

# ExVivo tissue mask (makesegfromvol defaults — same TakeOutBack core as
# ResampleToDay3End.py's build_body_mask, plus a largest-connected-component
# filter that drops the bright edge artifacts ExVivo scans sometimes have)
RETUNE_EXVIVO_MASK     = True
EXVIVO_MASK_THRESH     = 70
EXVIVO_MASK_MASKTHRESH = 160
EXVIVO_MASK_KERNSIZE   = 10

FLOAT32_EXACT_INT_CEILING = 2 ** 24  # largest integer float32 can represent exactly

# Pad amounts (voxels, isotropic) to sweep when checking how close padding gets us
# to the float32 ceiling. 0.5mm voxel spacing, so e.g. 20 voxels = 10mm.
PAD_VOXELS_TO_TEST = [0, 5, 10, 15, 20, 25, 30]

# Pad amount actually used to build/save the mask below.
PAD_VOXELS = 20


def _load_transform(path):
    """Load a Slicer .h5 transform via temp copy to sidestep HDF5 locking (same pattern as ResampleToDay3End.py)."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        return sitk.ReadTransform(tmp_path)
    finally:
        os.unlink(tmp_path)


def tune_exvivo_mask_interactively(vol_arr, thresh, maskthresh, kernsize):
    """Same interactive accept/retune loop as ResampleToDay3End.py's tune_mask_interactively,
    swapped to makesegfromvol (adds largest-connected-component filtering on top of the
    same TakeOutBack threshold+blur core, to reject isolated bright edge artifacts)."""
    while True:
        mask = makesegfromvol(vol_arr, thresh, maskthresh, kernsize)
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


def pad_mask(mask, pad_voxels):
    """Isotropic dilation of an arbitrary (possibly amorphous) 3D binary mask by
    pad_voxels, using a true Euclidean distance transform rather than a discrete
    structuring element — works the same regardless of the mask's shape/topology."""
    if pad_voxels <= 0:
        return mask
    dist_to_mask = distance_transform_edt(~mask)
    return dist_to_mask <= pad_voxels


def sweep_pad_counts(mask, pad_voxels_list):
    """Distance transform computed once; sweeping many pad radii off of it is then
    cheap, rather than re-running a full dilation per candidate radius."""
    dist_to_mask = distance_transform_edt(~mask)
    n_total = mask.size
    results = []
    for pad in pad_voxels_list:
        n = int((dist_to_mask <= pad).sum()) if pad > 0 else int(mask.sum())
        results.append((pad, n))
        pct_grid    = 100 * n / n_total
        pct_ceiling = 100 * n / FLOAT32_EXACT_INT_CEILING
        status = 'OK' if n < FLOAT32_EXACT_INT_CEILING else 'OVER CEILING'
        print(f"  pad={pad:3d} vox: {n:>10,} voxels  ({pct_grid:5.1f}% of grid, "
              f"{pct_ceiling:5.1f}% of float32 ceiling)  [{status}]")
    return results


def make_and_register_voxel_vol(RabbitID, Block, RabbitFolder, end_fixed='BlockFace'):
    paths    = find_all_the_paths(RabbitID, Block, RabbitFolder, 'InVivo')
    ref_path = str(paths['Moving_FilePath'])
    # paths['Fixed_FilePath']/['RegFold'] here are already the InVivo->ExVivo pair —
    # no separate find_all_the_paths(..., 'ExVivo') call needed.
    exvivo_path    = str(paths['Fixed_FilePath'])
    slicer_h5_path = next(p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._"))

    # --- Mask ExVivo to just tissue ---
    exvivo_nib = nib.load(exvivo_path)
    exvivo_arr = exvivo_nib.get_fdata(dtype=np.float32)
    if RETUNE_EXVIVO_MASK:
        exvivo_mask, *_ = tune_exvivo_mask_interactively(
            exvivo_arr, EXVIVO_MASK_THRESH, EXVIVO_MASK_MASKTHRESH, EXVIVO_MASK_KERNSIZE
        )
    else:
        exvivo_mask = makesegfromvol(exvivo_arr, EXVIVO_MASK_THRESH, EXVIVO_MASK_MASKTHRESH, EXVIVO_MASK_KERNSIZE)

    # --- Invert the InVivo<->ExVivo affine and warp the ExVivo mask into InVivo space ---
    # sitk.Resample's transform arg always maps OUTPUT(reference) space -> INPUT(moving)
    # space, regardless of what the transform file is named. This h5 is used elsewhere as
    # (moving=InVivo, fixed=ExVivo), so as-loaded it already maps ExVivo -> InVivo. We want
    # an InVivo-shaped output grid sampling the ExVivo mask, i.e. the opposite direction —
    # hence GetInverse() here.
    fwd_transform = _load_transform(slicer_h5_path)
    assert fwd_transform.GetName() in ('AffineTransform', 'MatrixOffsetTransformBase'), \
        f"Expected a bare AffineTransform, got {fwd_transform.GetName()} — this inversion approach assumes no embedded deformation field."
    inv_affine = sitk.AffineTransform(fwd_transform).GetInverse()

    invivo_ref_sitk  = sitk.ReadImage(ref_path)  # InVivo grid — defines the output space
    exvivo_mask_sitk = nib_to_sitk(exvivo_mask.astype(np.float32), exvivo_nib.affine)

    invivo_mask_sitk = sitk.Resample(
        exvivo_mask_sitk, invivo_ref_sitk, inv_affine, sitk.sitkNearestNeighbor, 0.0
    )
    invivo_mask = sitk.GetArrayFromImage(invivo_mask_sitk).transpose(2, 1, 0) > 0.5

    n_total = int(np.prod(nib.load(ref_path).shape[:3]))
    print(f"ExVivo-derived InVivo-space tissue mask, before padding: "
          f"{int(invivo_mask.sum()):,} / {n_total:,} voxels")

    # --- Save the padded mask for visual QC against the InVivo structural ---
    padded_mask = pad_mask(invivo_mask, PAD_VOXELS)
    proc_dir = os.path.join(paths['RegDataProc'], 'Voxel_Maps')
    os.makedirs(proc_dir, exist_ok=True)
    ref = nib.load(ref_path)
    mask_out_path = os.path.join(proc_dir, f'exvivo_derived_invivo_mask_pad{PAD_VOXELS}.nii.gz')
    nib.save(nib.Nifti1Image(padded_mask.astype(np.uint8), ref.affine), mask_out_path)
    print(f"Padded mask (pad={PAD_VOXELS}) saved → {mask_out_path}")
    print(f"  {int(padded_mask.sum()):,} voxels — load alongside {ref_path} in Slicer to check coverage")

    # --- Build compact unique labels over the padded mask ---
    # coords[id-1] == (i,j,k) for label value `id` — this lookup table is what makes
    # the compact IDs decodable later; the label volume alone can't be inverted by
    # arithmetic the way the old MOD-based scheme could.
    coords = np.argwhere(padded_mask).astype(np.int32)
    n_labels = len(coords)
    print(f"Compact unique labels: {n_labels:,} (id 1..{n_labels:,}, 0 = background) — "
          f"{'OK' if n_labels < FLOAT32_EXACT_INT_CEILING else 'OVER float32 CEILING'}")

    labels = np.zeros(padded_mask.shape, dtype=np.float32)
    labels[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(1, n_labels + 1, dtype=np.float32)

    lookup_path = os.path.join(proc_dir, f'voxel_tile_vol_unique_pad{PAD_VOXELS}_id_lookup.npy')
    np.save(lookup_path, coords)
    print(f"id -> (i,j,k) lookup table saved → {lookup_path}")

    voxel_vol_path = os.path.join(proc_dir, f'voxel_tile_vol_unique_pad{PAD_VOXELS}.nii.gz')
    nib.save(nib.Nifti1Image(labels, ref.affine), voxel_vol_path)
    print(f"Voxel label volume saved → {voxel_vol_path}")

    # --- Register InVivo → end_fixed (nearest-neighbour to preserve label values) ---
    print(f"Registering InVivo → {end_fixed}...")
    result, affine = MultiStepReg(
        RabbitID, Block, RabbitFolder,
        'InVivo', end_fixed,
        interpolation='nearest',
        moving_path=voxel_vol_path,
        save=False,
    )

    end_label = (end_fixed
                 .replace('ExVivoBlock', f'ExVivoBlock{Block:02d}')
                 .replace('BlockFace',   f'Block{Block:02d}'))
    out_path = os.path.join(proc_dir, f"voxel_tile_vol_unique_pad{PAD_VOXELS}_RegTo{end_label}.nii.gz")
    nib.save(nib.Nifti1Image(result, affine), out_path)
    print(f"Registered voxel volume saved → {out_path}")
    print(f"Decode: nib.load('{out_path}') voxel value v (v>0) -> "
          f"np.load('{lookup_path}')[int(v) - 1] gives the original InVivo (i,j,k)")

    # --- Same InVivo -> end_fixed registration via ComposedTransformReg's single-pass
    # composed field (one resample instead of one per stage) — saved alongside the
    # MultiStepReg output above, same naming convention, so the two can be loaded
    # side by side in Slicer to check whether voxel-fragmentation is reduced. ---
    print(f"Registering InVivo → {end_fixed} via composed transform chain...")
    composed_result, composed_affine = ComposedReg(
        RabbitID, Block, RabbitFolder,
        'InVivo', end_fixed,
        interpolation='nearest',
        moving_path=voxel_vol_path,
        save=False,
    )
    composed_out_path = os.path.join(proc_dir, f"voxel_tile_vol_unique_pad{PAD_VOXELS}_ComposedRegTo{end_label}.nii.gz")
    nib.save(nib.Nifti1Image(composed_result, composed_affine), composed_out_path)
    print(f"Composed-registered voxel volume saved → {composed_out_path}")

    return out_path, composed_out_path, lookup_path


if __name__ == '__main__':
    make_and_register_voxel_vol(RABBIT_ID, BLOCK, RABBIT_FOLDER, END_FIXED)
