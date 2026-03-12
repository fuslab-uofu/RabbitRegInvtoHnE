#Import libraries-
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import numpy as np
import nibabel as nib
from ApplyTranforms import *
from RabbitPathFInder import *
from NiftyTransforms import makesegfromvol, match_histograms, bin_intensities, DimsDivFour
from datetime import datetime
import matplotlib.pyplot as plt
import glob
from PIL import Image

def PrepForLandMarker(movingimpath, fixedimpath, SlicerTPath, output_dir,
                      moving_stage, fixed_stage, Block=None, processing=None,
                      fixed_output_dir=None):
    """
    Part 1: Apply Slicer rigid transform, resample moving into fixed space,
    canonicalize, and save as .nii.gz ready for LandMarker.

    processing='mask_and_histmatch': additionally computes a tissue mask from the
    fixed volume, applies it to both volumes, and histogram-matches the resampled
    moving to the masked fixed. Use for intensity-based registration stages
    (e.g. ExVivo->ExVivoBlock).
    """
    warn_if_oblique(movingimpath)
    warn_if_oblique(fixedimpath)

    # Resample moving into fixed image space
    resampled = ApplySlicerTransform(str(movingimpath), str(fixedimpath), str(SlicerTPath))

    # Canonicalize to match LandMarker's convention
    fixed_nib = nib.load(str(fixedimpath))
    fixed_nib_canonical = nib.as_closest_canonical(fixed_nib)
    resampled = np.asanyarray(
        nib.as_closest_canonical(nib.Nifti1Image(resampled, fixed_nib.affine)).dataobj
    ).astype(np.float32)

    block_suffix = f"{Block:02d}" if Block is not None else ""
    def label(stage):
        return f"{stage}{block_suffix}" if "Block" in stage else stage

    os.makedirs(output_dir, exist_ok=True)

    if processing == 'mask_and_histmatch':
        fixed_array = fixed_nib_canonical.get_fdata().astype(np.float32)

        # Normalize both volumes to [0, 255] before masking and matching
        resampled_norm = resampled / np.max(resampled) * 255 if np.max(resampled) > 0 else resampled
        fixed_norm = fixed_array / np.max(fixed_array) * 255 if np.max(fixed_array) > 0 else fixed_array

        mask = makesegfromvol(fixed_norm)

        resampled_masked = resampled_norm * mask
        fixed_masked = fixed_norm * mask

        resampled_matched = match_histograms(resampled_masked, fixed_masked)

        resampled_binned = bin_intensities(resampled_matched)
        fixed_binned     = bin_intensities(fixed_masked)

        fixed_dir = fixed_output_dir if fixed_output_dir is not None else output_dir
        os.makedirs(fixed_dir, exist_ok=True)

        resampled_out     = os.path.join(output_dir, f"{label(moving_stage)}ResampledTo{label(fixed_stage)}.nii.gz")
        moving_out        = os.path.join(output_dir, f"{label(moving_stage)}MaskedTo{label(fixed_stage)}.nii.gz")
        moving_binned_out = os.path.join(output_dir, f"{label(moving_stage)}MaskedTo{label(fixed_stage)}_binned.nii.gz")
        fixed_out         = os.path.join(fixed_dir,  f"{label(fixed_stage)}Masked.nii.gz")
        fixed_binned_out  = os.path.join(fixed_dir,  f"{label(fixed_stage)}Masked_binned.nii.gz")
        nib.save(nib.Nifti1Image(resampled,                                          fixed_nib_canonical.affine), resampled_out)
        nib.save(nib.Nifti1Image(DimsDivFour(resampled_matched).astype(np.float32), fixed_nib_canonical.affine), moving_out)
        nib.save(nib.Nifti1Image(DimsDivFour(resampled_binned).astype(np.float32),  fixed_nib_canonical.affine), moving_binned_out)
        nib.save(nib.Nifti1Image(DimsDivFour(fixed_masked).astype(np.float32),      fixed_nib_canonical.affine), fixed_out)
        nib.save(nib.Nifti1Image(DimsDivFour(fixed_binned).astype(np.float32),      fixed_nib_canonical.affine), fixed_binned_out)
        print(f"Saved resampled moving to {resampled_out}")
        print(f"Saved masked+matched moving to {moving_out}")
        print(f"Saved masked+matched+binned moving to {moving_binned_out}")
        print(f"Saved masked fixed to {fixed_out}")
        print(f"Saved masked+binned fixed to {fixed_binned_out}")
    else:
        output_path = os.path.join(output_dir, f"{label(moving_stage)}ResampledTo{label(fixed_stage)}.nii.gz")
        nib.save(nib.Nifti1Image(resampled, fixed_nib_canonical.affine), output_path)
        print(f"Saved to {output_path}")

    return resampled, fixed_nib_canonical.affine


def BlockFaceToNifti(blockface_folder):
    """
    Convert a BlockFace folder's CroppedImages TIFFs to a 4x-downsampled greyscale NIfTI.
    Output saved as greyscale_downsampled.nii.gz in blockface_folder.
    Skips if greyscale_downsampled.nii.gz already exists.
    Output spacing: 0.088mm XY (4 * 0.022mm), 0.05mm Z. Orientation: LPS (L, P, S).
    """
    output_path = os.path.join(blockface_folder, 'greyscale_downsampled.nii.gz')
    if os.path.exists(output_path):
        print(f"greyscale_downsampled.nii.gz already exists, skipping.")
        return output_path

    cropped_images_dir = os.path.join(blockface_folder, 'CroppedImages')
    files = sorted([f for f in os.listdir(cropped_images_dir) if f.endswith('.tiff') and not f.startswith('._')])

    slices = []
    for f in files:
        img = np.array(Image.open(os.path.join(cropped_images_dir, f)))
        r, g, b = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
        grey = 255 - (0.3 * r + 0.35 * g + 0.35 * b)
        grey = np.where(img[:, :, 0] == 0, 0, grey)
        slices.append(grey.astype(np.float32))

    # Stack (Z, H, W) then transpose to NIfTI (X=W, Y=H, Z)
    vol = np.stack(slices, axis=0)        # (Z, H, W)
    vol = np.transpose(vol, (2, 1, 0))    # (W, H, Z) = (X, Y, Z)

    # 4x downsample in X and Y
    vol_ds = vol[::4, ::4, :]

    # XY spacing 0.088mm, Z spacing 0.05mm, LPS orientation
    affine = np.array([
        [-0.088,  0.,     0.,    0.],
        [ 0.,    -0.088,  0.,    0.],
        [ 0.,     0.,     0.05,  0.],
        [ 0.,     0.,     0.,    1.],
    ])

    nib.save(nib.Nifti1Image(vol_ds, affine), output_path)
    print(f"Saved to {output_path}")
    return output_path


def MultiStepPrepForLandMarker(RabbitID, Block, RabbitFolder):
    PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]
    for i in range(len(PROGRESSION) - 1):
        moving_key = PROGRESSION[i]
        fixed_key  = PROGRESSION[i + 1]
        paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)

        # For ExVivoBlock→BlockFace, ensure greyscale NIfTI exists before proceeding
        if moving_key == "ExVivoBlock":
            paths["Fixed_FilePath"] = BlockFaceToNifti(paths["Fixed_Folder"])

        SlicerTPath = next((p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._")), None)

        PrepForLandMarker(
            movingimpath     = paths['Moving_FilePath'],
            fixedimpath      = paths['Fixed_FilePath'],
            SlicerTPath      = SlicerTPath,
            output_dir       = paths['RegDataProc'],
            moving_stage     = moving_key,
            fixed_stage      = fixed_key,
            Block            = Block,
            processing       = 'mask_and_histmatch' if moving_key == 'ExVivo' else None,
            fixed_output_dir = os.path.join(paths['Fixed_Folder'], 'RegDataProc') if moving_key == 'ExVivo' else None,
        )


#Set which Rabbit and Block we want, RabbitData is where it all lives, folder structure matters here->
RabbitFolder='/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID="R24-240"
Block = 1

MultiStepPrepForLandMarker(RabbitID, Block, RabbitFolder)
