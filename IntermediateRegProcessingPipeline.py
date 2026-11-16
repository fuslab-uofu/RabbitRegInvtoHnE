#Import libraries-
import os
import re
import SimpleITK as sitk
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import numpy as np
import nibabel as nib
from ApplyTransforms import *
from RabbitPathFinder import *
from MaskAndHistMatch import mask_and_histmatch_block, resolve_cropped_path, DEFAULT_THRESH, DEFAULT_MASKTHRESH, DEFAULT_KERNSIZE
from datetime import datetime
import matplotlib.pyplot as plt
import glob
from PIL import Image

def PrepForLandMarker(movingimpath, fixedimpath, SlicerTPath, output_dir,
                      moving_stage, fixed_stage, Block=None, interpolator=sitk.sitkLinear):
    """
    Apply Slicer rigid transform, resample moving into fixed space,
    canonicalize, and save as .nii.gz ready for LandMarker.
    """
    warn_if_oblique(movingimpath)
    warn_if_oblique(fixedimpath)

    # Resample moving into fixed image space
    resampled = ApplySlicerTransform(str(movingimpath), str(fixedimpath), str(SlicerTPath), interpolator=interpolator)

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


def PrepInVivoToExVivo(RabbitID, RabbitFolder, Block):
    """
    Run the InVivo->ExVivo resample once per rabbit. This stage's inputs and output
    path don't depend on Block, so it only needs to run once no matter how many
    blocks the rabbit has -- Block is only passed through to satisfy
    find_all_the_paths()'s signature, it has no effect on this stage's paths.
    Always overwrites the output, so a redone Slicer affine takes effect.
    """
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, "InVivo")
    SlicerTPath = find_single_transform(paths['RegFold'], ".h5")

    PrepForLandMarker(
        movingimpath = paths['Moving_FilePath'],
        fixedimpath  = paths['Fixed_FilePath'],
        SlicerTPath  = SlicerTPath,
        output_dir   = paths['RegDataProc'],
        moving_stage = "InVivo",
        fixed_stage  = "ExVivo",
    )


def PrepExVivoToExVivoBlock(RabbitID, Block, RabbitFolder, thresh, maskthresh, kernsize, interactive=True):
    """
    Run ExVivo->ExVivoBlock (resample, then interactive mask+histogram-match via
    MaskAndHistMatch.mask_and_histmatch_block), for one block. Returns the
    (possibly retuned) thresh, maskthresh, kernsize so the caller can carry them
    forward as the starting point for the next block.

    Devs haven't flushed this out as its own clean standalone stage yet -- this
    is still just the ExVivo->ExVivoBlock leg lifted out of the combined
    ExVivo->ExVivoBlock->BlockFace loop, not a redesigned stage.
    """
    moving_key, fixed_key = "ExVivo", "ExVivoBlock"
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)
    SlicerTPath = find_single_transform(paths['RegFold'], ".h5")

    PrepForLandMarker(
        movingimpath = paths['Moving_FilePath'],
        fixedimpath  = paths['Fixed_FilePath'],
        SlicerTPath  = SlicerTPath,
        output_dir   = paths['RegDataProc'],
        moving_stage = moving_key,
        fixed_stage  = fixed_key,
        Block        = Block,
    )

    thresh, maskthresh, kernsize = mask_and_histmatch_block(
        RabbitID, Block, RabbitFolder, thresh, maskthresh, kernsize, interactive=interactive
    )
    return thresh, maskthresh, kernsize


def PrepExVivoBlockToBlockFace(RabbitID, Block, RabbitFolder):
    """
    Run ExVivoBlock->BlockFace (resample only, no landmark/mask step), for one
    block at a time.
    """
    moving_key, fixed_key = "ExVivoBlock", "BlockFace"
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)
    # Ensure greyscale NIfTI exists before proceeding
    paths["Fixed_FilePath"] = BlockFaceToNifti(paths["Fixed_Folder"])

    SlicerTPath = find_single_transform(paths['RegFold'], ".h5")

    PrepForLandMarker(
        movingimpath = paths['Moving_FilePath'],
        fixedimpath  = paths['Fixed_FilePath'],
        SlicerTPath  = SlicerTPath,
        output_dir   = paths['RegDataProc'],
        moving_stage = moving_key,
        fixed_stage  = fixed_key,
        Block        = Block,
    )


def discover_ready_blocks(RabbitID, RabbitFolder):
    """Every BlockNN subfolder under this rabbit's ExVivo_MRBlocked/ directory that
    has both prerequisites for a first ExVivo->ExVivoBlock run: the raw Cropped scan,
    and a Slicer .h5 transform in ExVivo_MR/BlockNNReg/RegTransforms/. Skips (with a
    message) any block missing either, rather than discovering a block this pipeline
    can't actually process yet."""
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
        slicer_path = next((p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._")), None)
        if cropped_path is None:
            print(f"  Skipping Block{block_suffix}: no Cropped scan found")
            continue
        if slicer_path is None:
            print(f"  Skipping Block{block_suffix}: no Slicer .h5 transform found")
            continue
        blocks.append(block)
    return blocks


def RunRegistrationPipelineForRabbit(RabbitID, Blocks, RabbitFolder, interactive=True):
    if not Blocks:
        print(f"No ready blocks for {RabbitID}, nothing to do.")
        return

    PrepInVivoToExVivo(RabbitID, RabbitFolder, Blocks[0])

    thresh, maskthresh, kernsize = DEFAULT_THRESH, DEFAULT_MASKTHRESH, DEFAULT_KERNSIZE
    for Block in Blocks:
        thresh, maskthresh, kernsize = PrepExVivoToExVivoBlock(
            RabbitID, Block, RabbitFolder, thresh, maskthresh, kernsize, interactive=interactive
        )
        PrepExVivoBlockToBlockFace(RabbitID, Block, RabbitFolder)


def RunAllReadyBlocks(RabbitID, Block, RabbitFolder):
    """First-time prep of a whole rabbit: InVivo->ExVivo once, then the two block-level
    stages for every block that's ready. Block is unused -- blocks are discovered."""
    Blocks = discover_ready_blocks(RabbitID, RabbitFolder)
    print(f"Found {len(Blocks)} block(s) ready to process for {RabbitID}: {Blocks}")
    RunRegistrationPipelineForRabbit(RabbitID, Blocks, RabbitFolder)


#Each stage preps one registration step's moving volume for LandMarker.
STAGES = {
    "InVivoToExVivo":         lambda RabbitID, Block, RabbitFolder: PrepInVivoToExVivo(RabbitID, RabbitFolder, Block),
    "ExVivoToExVivoBlock":    lambda RabbitID, Block, RabbitFolder: PrepExVivoToExVivoBlock(
                                  RabbitID, Block, RabbitFolder, DEFAULT_THRESH, DEFAULT_MASKTHRESH, DEFAULT_KERNSIZE),
    "ExVivoBlockToBlockFace": PrepExVivoBlockToBlockFace,
    "AllReadyBlocks":         RunAllReadyBlocks,
}


def run_stage(stage, RabbitID, Block, RabbitFolder):
    if stage not in STAGES:
        raise ValueError(f"Unknown STAGE '{stage}'. Choose one of: {', '.join(STAGES)}")
    STAGES[stage](RabbitID, Block, RabbitFolder)


#Set which Rabbit we want, RabbitData is where it all lives, folder structure matters here->
RabbitFolder='/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID="R24-103"
Block = 4  # not used by InVivoToExVivo or AllReadyBlocks

#Which step to re-prep for LandMarker. Only rerun after redoing that step's Slicer affine->
# "InVivoToExVivo", "ExVivoToExVivoBlock", "ExVivoBlockToBlockFace",
# or "AllReadyBlocks" for first-time prep of every ready block of a rabbit.
STAGE = "ExVivoToExVivoBlock"

if __name__ == '__main__':
    run_stage(STAGE, RabbitID, Block, RabbitFolder)
