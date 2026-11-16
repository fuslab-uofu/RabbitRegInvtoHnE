"""
Masking + histogram-matching for the ExVivo->ExVivoBlock intensity-based registration
stage, pulled out of IntermediateRegProcessingPipeline.py's
PrepForLandMarker(processing='mask_and_histmatch') so it can be rerun directly against
the already-resampled volumes -- no need to redo the (slower) Slicer rigid resample --
and reapplied in bulk across previously-processed rabbits/blocks once the masking
approach itself is reworked.

Explicit expected filenames are used for the moving/fixed inputs rather than
find_all_the_paths()'s find_nii() (which just returns the first *.nii.gz it finds in
the folder) -- a block's folder isn't guaranteed to have only one nii.gz in it.

Masking is interactively tunable (same pattern as MakeVoxelVolV2.py's
tune_exvivo_mask_interactively) instead of silently using makesegfromvol()'s generic
defaults -- unlike that function, this also returns the accepted thresh/maskthresh/
kernsize so a batch run can carry them forward block-to-block and log what was used.
"""
import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from RabbitPathFinder import find_all_the_paths
from NiftyTransforms import makesegfromvol, match_histograms, bin_intensities, DimsDivFour

DEFAULT_THRESH = 70
DEFAULT_MASKTHRESH = 160
DEFAULT_KERNSIZE = 10


def resolve_cropped_path(fixed_folder):
    """The raw ExVivo-block scan's filename convention varies a lot across rabbits --
    seen so far: ExVivoBlock{NN}Cropped (R24-058), ExVMR_Block{NN}Cropped (R23-055),
    and on R24-103 alone: ExVivoMR_Block4 (no zero-pad, no "Cropped"),
    ExVivoMR_Block05 (zero-padded, still no "Cropped"), and ExVMR_Block06Cropped --
    three different conventions in the same rabbit. Chasing each new naming pattern
    doesn't generalize, so this relies on a structural invariant instead: the raw
    scan is the only *.nii(.gz) file sitting directly in this block's own top-level
    ExVivo_MRBlocked/Block{NN}/ folder -- every derived/processed output lives one
    level down in a Reg* subfolder, not at that top level. Downstream-looking names
    (containing "resampled"/"masked"/"binned") are excluded in case more than one
    file is still sitting there, e.g. R23-055/Block05's stray _resampled copy.
    Prefers .nii.gz over plain .nii when more than one real candidate remains.
    Returns None (not raise) so callers doing discovery can just skip a block that
    isn't ready instead of crashing a whole batch run partway through."""
    exclude_markers = ('resampled', 'masked', 'binned')
    candidates = sorted(
        f for f in os.listdir(fixed_folder)
        if not f.startswith('._')
        and os.path.isfile(os.path.join(fixed_folder, f))
        and (f.endswith('.nii') or f.endswith('.nii.gz'))
        and not any(marker in f.lower() for marker in exclude_markers)
    )
    if not candidates:
        return None
    gz_candidates = [f for f in candidates if f.endswith('.nii.gz')]
    chosen = gz_candidates[0] if gz_candidates else candidates[0]
    if len(candidates) > 1:
        print(f"  Note: multiple raw scan candidates in {fixed_folder}: {candidates} -- using {chosen}")
    return os.path.join(fixed_folder, chosen)


def discover_blocks(RabbitID, RabbitFolder):
    """Every BlockNN subfolder under this rabbit's ExVivo_MRBlocked/ directory that
    actually has both prerequisite files ready: the raw Cropped scan, and the
    already-resampled moving volume from a prior IntermediateRegProcessingPipeline
    run. Skips (with a message) any block missing either, rather than discovering a
    block this script can't actually process yet."""
    blocked_dir = os.path.join(RabbitFolder, RabbitID, 'ExVivo_MRBlocked')
    blocks = []
    for entry in sorted(os.listdir(blocked_dir)):
        m = re.fullmatch(r'Block(\d+)', entry)
        if not m:
            continue
        block = int(m.group(1))
        block_suffix = f"{block:02d}"
        paths = find_all_the_paths(RabbitID, block, RabbitFolder, 'ExVivo')

        cropped_path = resolve_cropped_path(paths['Fixed_Folder'])
        resampled_path = os.path.join(paths['RegDataProc'], f"ExVivoResampledToExVivoBlock{block_suffix}.nii.gz")
        if cropped_path is None:
            print(f"  Skipping Block{block_suffix}: no Cropped scan found")
            continue
        if not os.path.exists(resampled_path):
            print(f"  Skipping Block{block_suffix}: no resampled moving volume yet (run IRPP first?)")
            continue
        blocks.append(block)
    return blocks


AXIS_NAMES = ('x', 'y', 'z')
# For each sliced axis, the (row, col) array axes of the resulting 2D plane -- e.g.
# slicing axis 0 (x) leaves a (y, z)-shaped plane with y as rows, z as columns.
_PLANE_AXES = ((1, 2), (0, 2), (0, 1))


def _draw_origin_arrows(ax, row_axis_name, col_axis_name, shape):
    """Draw two arrows from the slice's origin corner (0, 0) showing which
    in-plane direction is +row and +col, so thresh/maskthresh formulas like
    '10+4x' can be written with the right sign."""
    row_len = shape[0] * 0.15
    col_len = shape[1] * 0.15
    ax.annotate('', xy=(0, row_len), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(0, row_len, f' +{row_axis_name}', color='red', fontsize=8, va='top')
    ax.annotate('', xy=(col_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(col_len, 0, f' +{col_axis_name}', color='red', fontsize=8, va='bottom')


def tune_mask_interactively(vol_arr, thresh, maskthresh, kernsize, n_slices=5):
    """Same interactive accept/retune loop as MakeVoxelVolV2.py's
    tune_exvivo_mask_interactively, but shows n_slices evenly-spaced slices per axis
    (that function only showed the single center slice per axis) so boundary/edge
    regions of the mask are visible too, not just the middle.

    thresh and maskthresh may be a plain number, or a string formula in x, y, z (the
    voxel index along each array axis) for a directional threshold, e.g. '10+4x' to
    compensate for a coil-inhomogeneity gradient along axis 0 -- see
    resolve_threshold_field() in NiftyTransforms.py. Each axis-group's first subplot
    shows an origin arrow for the two in-plane axes so formula signs can be picked
    correctly.
    """
    while True:
        mask = makesegfromvol(vol_arr, thresh, maskthresh, kernsize)
        pic = np.array(vol_arr / np.max(vol_arr) * 255).astype(np.uint8)

        axis_slicers = [
            lambda p, m, i: (p[i, :, :], m[i, :, :]),
            lambda p, m, i: (p[:, i, :], m[:, i, :]),
            lambda p, m, i: (p[:, :, i], m[:, :, i]),
        ]
        fig, axs = plt.subplots(6, n_slices, figsize=(2 * n_slices, 9))
        for axis in range(3):
            idxs = np.linspace(0, vol_arr.shape[axis] - 1, n_slices).astype(int)
            row_axis, col_axis = _PLANE_AXES[axis]
            for col, i in enumerate(idxs):
                sl_img, sl_mask = axis_slicers[axis](pic, mask, i)
                axs[2 * axis, col].imshow(sl_img, cmap='gray')
                axs[2 * axis, col].set_title(f'axis {AXIS_NAMES[axis]}, slice {i}', fontsize=8)
                axs[2 * axis + 1, col].imshow(sl_img * sl_mask, cmap='gray')
                if col == 0:
                    _draw_origin_arrows(axs[2 * axis, col], AXIS_NAMES[row_axis], AXIS_NAMES[col_axis], sl_img.shape)
                axs[2 * axis, col].axis('off')
                axs[2 * axis + 1, col].axis('off')
        plt.suptitle(f'thresh={thresh}  maskthresh={maskthresh}  kernsize={kernsize}'
                     f'\nPer axis: top=original, bottom=masked')
        plt.tight_layout()
        plt.show()
        if input('Accept mask? (y/n): ').strip().lower() == 'y':
            return mask, thresh, maskthresh, kernsize
        val = input(f'  thresh [{thresh}]: ').strip()
        thresh = val if val else thresh
        val = input(f'  maskthresh [{maskthresh}]: ').strip()
        maskthresh = val if val else maskthresh
        val = input(f'  kernsize [{kernsize}]: ').strip()
        kernsize = int(val) if val else kernsize


def mask_and_histmatch_block(RabbitID, Block, RabbitFolder, thresh=DEFAULT_THRESH,
                              maskthresh=DEFAULT_MASKTHRESH, kernsize=DEFAULT_KERNSIZE,
                              interactive=True, make_mask=True):
    """make_mask=False skips masking/histogram-matching entirely and instead loads
    the already-saved masked(+matched) volumes from a prior make_mask=True run, then
    only recomputes and overwrites the *_binned outputs -- for rerunning binning
    (e.g. after a bin_intensities change) without redoing the interactive masking
    step. thresh/maskthresh/kernsize are unused in that case."""
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, 'ExVivo')
    block_suffix = f"{Block:02d}"

    fixed_path = resolve_cropped_path(paths['Fixed_Folder'])
    if fixed_path is None:
        raise FileNotFoundError(f"No raw scan (*.nii/*.nii.gz) found in {paths['Fixed_Folder']}")

    fixed_nib_canonical = nib.as_closest_canonical(nib.load(fixed_path))

    fixed_output_dir = os.path.join(paths['Fixed_Folder'], 'RegDataProc')
    os.makedirs(paths['RegDataProc'], exist_ok=True)
    os.makedirs(fixed_output_dir, exist_ok=True)

    moving_out = os.path.join(paths['RegDataProc'], f"ExVivoMaskedToExVivoBlock{block_suffix}.nii.gz")
    moving_binned_out = os.path.join(paths['RegDataProc'], f"ExVivoMaskedToExVivoBlock{block_suffix}_binned.nii.gz")
    fixed_out = os.path.join(fixed_output_dir, f"ExVivoBlock{block_suffix}Masked.nii.gz")
    fixed_binned_out = os.path.join(fixed_output_dir, f"ExVivoBlock{block_suffix}Masked_binned.nii.gz")

    if make_mask:
        moving_path = os.path.join(paths['RegDataProc'], f"ExVivoResampledToExVivoBlock{block_suffix}.nii.gz")
        # Already resampled into the fixed volume's canonical space by
        # PrepForLandMarker, so no re-canonicalization needed here -- just load it.
        resampled = nib.load(moving_path).get_fdata().astype(np.float32)
        fixed_array = fixed_nib_canonical.get_fdata().astype(np.float32)

        # Normalize both volumes to [0, 255] before masking and matching
        resampled_norm = resampled / np.max(resampled) * 255 if np.max(resampled) > 0 else resampled
        fixed_norm = fixed_array / np.max(fixed_array) * 255 if np.max(fixed_array) > 0 else fixed_array

        if interactive:
            mask, thresh, maskthresh, kernsize = tune_mask_interactively(fixed_norm, thresh, maskthresh, kernsize)
        else:
            mask = makesegfromvol(fixed_norm, thresh, maskthresh, kernsize)

        print(f"[{RabbitID} Block{block_suffix}] mask covers {100 * mask.mean():.1f}% of the volume "
              f"({mask.sum()}/{mask.size} voxels) -- thresh={thresh} maskthresh={maskthresh} kernsize={kernsize}")

        resampled_masked = resampled_norm * mask
        fixed_masked = fixed_norm * mask

        resampled_matched = match_histograms(resampled_masked, fixed_masked)

        nib.save(nib.Nifti1Image(DimsDivFour(resampled_matched).astype(np.float32), fixed_nib_canonical.affine), moving_out)
        nib.save(nib.Nifti1Image(DimsDivFour(fixed_masked).astype(np.float32), fixed_nib_canonical.affine), fixed_out)
        print(f"[{RabbitID} Block{block_suffix}] Saved masked+matched moving to {moving_out}")
        print(f"[{RabbitID} Block{block_suffix}] Saved masked fixed to {fixed_out}")
    else:
        print(f"[{RabbitID} Block{block_suffix}] make_mask=False -- loading existing masked volumes from disk")
        resampled_matched = nib.load(moving_out).get_fdata().astype(np.float32)
        fixed_masked = nib.load(fixed_out).get_fdata().astype(np.float32)

    resampled_binned = bin_intensities(resampled_matched)
    fixed_binned = bin_intensities(fixed_masked)

    nib.save(nib.Nifti1Image(DimsDivFour(resampled_binned).astype(np.float32), fixed_nib_canonical.affine), moving_binned_out)
    nib.save(nib.Nifti1Image(DimsDivFour(fixed_binned).astype(np.float32), fixed_nib_canonical.affine), fixed_binned_out)

    print(f"[{RabbitID} Block{block_suffix}] Saved masked+matched+binned moving to {moving_binned_out}")
    print(f"[{RabbitID} Block{block_suffix}] Saved masked+binned fixed to {fixed_binned_out}")

    return thresh, maskthresh, kernsize


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID = 'R24-124'

    blocks = discover_blocks(RabbitID, RabbitFolder)
    print(f"Found {len(blocks)} block(s) ready to process for {RabbitID}: {blocks}")

    thresh, maskthresh, kernsize = DEFAULT_THRESH, DEFAULT_MASKTHRESH, DEFAULT_KERNSIZE
    for Block in blocks:
        # Carry the last-accepted params forward as the starting point for the next
        # block -- block-to-block masking params should be similar, so after the
        # first block or two this should mostly just mean hitting 'y'.
        thresh, maskthresh, kernsize = mask_and_histmatch_block(
            RabbitID, Block, RabbitFolder, thresh, maskthresh, kernsize, make_mask=True
        )
