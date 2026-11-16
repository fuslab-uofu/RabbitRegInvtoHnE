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

# ExVivo tissue mask- initial threshold, then convlutional filter applied with kernsize, then thresholded again-
RETUNE_EXVIVO_MASK     = True
EXVIVO_MASK_THRESH     = 70
EXVIVO_MASK_MASKTHRESH = 160
EXVIVO_MASK_KERNSIZE   = 10

FLOAT32_EXACT_INT_CEILING = 2 ** 24  # largest integer float32 can represent exactly

# Padding applied to make Ex Vivo Mask Overflow Tissue Region-
PAD_VOXELS = 20
# _________________

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
    #Accept/retune loop to tune_mask_interactively-
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
    #Pad mask using Euclidean distance from mask->
    if pad_voxels <= 0:
        return mask
    dist_to_mask = distance_transform_edt(~mask)
    #Distances converted to Booleans->
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
    # path finding and check if voxel volume exists already.
    paths    = find_all_the_paths(RabbitID, Block, RabbitFolder, 'InVivo')
    ref_path = str(paths['Moving_FilePath'])
    proc_dir = os.path.join(paths['RegDataProc'], 'Voxel_Maps')
    os.makedirs(proc_dir, exist_ok=True)
    voxel_vol_path = os.path.join(proc_dir, 'voxel_tile_vol.nii.gz')
    lookup_path    = os.path.join(proc_dir, 'voxel_tile_vol_id_lookup.npy')

    voxel_vol_exists = os.path.exists(voxel_vol_path)
    lookup_exists    = os.path.exists(lookup_path)

    # Mask + labels are built once per rabbit-
    if voxel_vol_exists and lookup_exists:
        print(f"Voxel volume + ID lookup already exist for this rabbit — reusing "
              f"{voxel_vol_path} instead of rebuilding, so existing voxel_id-keyed "
              f"outputs from other blocks stay valid. Delete both this file and "
              f"{lookup_path} together to force a full rebuild.")
    elif voxel_vol_exists != lookup_exists:
        raise FileNotFoundError(
            f"Found only one of the paired voxel-volume outputs (voxel_vol_path "
            f"exists={voxel_vol_exists}, lookup_path exists={lookup_exists}) — these "
            f"must be regenerated together. Delete whichever one remains and re-run."
        )
    else:
        exvivo_path    = str(paths['Fixed_FilePath'])
        slicer_h5_path = next(p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._"))

        # --- Mask Ex Vivo to just tissue ---
        exvivo_nib = nib.load(exvivo_path)
        exvivo_arr = exvivo_nib.get_fdata(dtype=np.float32)
        #Option to tweak parameters-
        if RETUNE_EXVIVO_MASK:
            exvivo_mask, *_ = tune_exvivo_mask_interactively(
                exvivo_arr, EXVIVO_MASK_THRESH, EXVIVO_MASK_MASKTHRESH, EXVIVO_MASK_KERNSIZE
            )
        else:
            exvivo_mask = makesegfromvol(exvivo_arr, EXVIVO_MASK_THRESH, EXVIVO_MASK_MASKTHRESH, EXVIVO_MASK_KERNSIZE)

        # Invert the InVivo<->ExVivo affine and warp the ExVivo mask into InVivo space
        fwd_transform = _load_transform(slicer_h5_path)
        #Check that its just an affine-
        assert fwd_transform.GetName() in ('AffineTransform', 'MatrixOffsetTransformBase'), \
            f"Expected a bare AffineTransform, got {fwd_transform.GetName()} — this inversion approach assumes no embedded deformation field."
        inv_affine = sitk.AffineTransform(fwd_transform).GetInverse()

        invivo_ref_sitk  = sitk.ReadImage(ref_path)  # InVivo grid — defines the output space
        exvivo_mask_sitk = nib_to_sitk(exvivo_mask.astype(np.float32), exvivo_nib.affine)

        #Mask resampled to in vivo space->
        invivo_mask_sitk = sitk.Resample(
            exvivo_mask_sitk, invivo_ref_sitk, inv_affine, sitk.sitkNearestNeighbor, 0.0
        )
        #Converted to array-
        invivo_mask = sitk.GetArrayFromImage(invivo_mask_sitk).transpose(2, 1, 0) > 0.5

        #Fraction of volume covered-
        n_total = int(np.prod(nib.load(ref_path).shape[:3]))
        print(f"ExVivo-derived InVivo-space tissue mask, before padding: "
              f"{int(invivo_mask.sum()):,} / {n_total:,} voxels")

        # Save the padded mask for visual QC against the InVivo structural
        padded_mask = pad_mask(invivo_mask, PAD_VOXELS)
        ref = nib.load(ref_path)
        mask_out_path = os.path.join(proc_dir, f'exvivo_derived_invivo_mask_pad{PAD_VOXELS}.nii.gz')
        nib.save(nib.Nifti1Image(padded_mask.astype(np.uint8), ref.affine), mask_out_path)
        print(f"Padded mask (pad={PAD_VOXELS}) saved → {mask_out_path}")
        print(f"  {int(padded_mask.sum()):,} voxels — load alongside {ref_path} in Slicer to check coverage")

        # --- Build compact unique labels over the padded mask ---
        # coords[id-1] == (i,j,k) for label value `id`
        #row of coords becomes the voxelID +1
        coords = np.argwhere(padded_mask).astype(np.int32)
        print(f"id -> (i,j,k) lookup table saving → {lookup_path}")
        np.save(lookup_path, coords)

        n_labels = len(coords)
        print(f"Compact unique labels: {n_labels:,} (id 1..{n_labels:,}, 0 = background) — "
              f"{'OK' if n_labels < FLOAT32_EXACT_INT_CEILING else 'OVER float32 CEILING'}")
        labels = np.zeros(padded_mask.shape, dtype=np.float32)
        #Labels applied to volume-
        labels[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(1, n_labels + 1, dtype=np.float32)

        print(f"Voxel label volume saving → {voxel_vol_path}")
        nib.save(nib.Nifti1Image(labels, ref.affine), voxel_vol_path)

    #Register Voxel Volume to Block Face->
    composed_result, composed_affine = ComposedReg(
        RabbitID, Block, RabbitFolder,
        'InVivo', end_fixed,
        interpolation='nearest',
        moving_path=voxel_vol_path,
        save=False,
    )

    end_label = (end_fixed
                 .replace('ExVivoBlock', f'ExVivoBlock{Block:02d}')
                 .replace('BlockFace',   f'Block{Block:02d}'))
    composed_out_path = os.path.join(proc_dir, f"voxel_tile_vol_ComposedRegTo{end_label}.nii.gz")
    nib.save(nib.Nifti1Image(composed_result, composed_affine), composed_out_path)
    print(f"Composed-registered voxel volume saved → {composed_out_path}")
    print(f"Decode: nib.load('{composed_out_path}') voxel value v (v>0) -> "
          f"np.load('{lookup_path}')[int(v) - 1] gives the original InVivo (i,j,k)")

    return composed_out_path, lookup_path


if __name__ == '__main__':
    make_and_register_voxel_vol(RABBIT_ID, BLOCK, RABBIT_FOLDER, END_FIXED)
