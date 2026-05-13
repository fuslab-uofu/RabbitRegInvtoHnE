#Import libraries-
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import re
import numpy as np
import nibabel as nib
from ApplyTransforms import *
from RabbitPathFinder import *
from datetime import datetime
import matplotlib.pyplot as plt
import glob

#Set which Rabbit and Block we want, RabbitData is where it all lives, folder structure matters here->
RabbitFolder='/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID="R23-055"
Block = 7

#If we're working from a directory->
RegDir="/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/InVMRDataSets/Day3End_Registered"

#Generalizing to be able to do multiple steps->
#Provide key for moving and final fixed volume. Options- ["InVivo", "ExVivo", "ExVivoBlock", "Blockface"]
def MultiStepReg(RabbitID, Block, RabbitFolder, MovingStart, EndFixed, interpolation='nearest', moving_path=None, save=True):
    PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]

    start_idx = PROGRESSION.index(MovingStart)
    end_idx   = PROGRESSION.index(EndFixed)

    if end_idx <= start_idx:
        print("The two volumes selected are not fit for registration")
        return

    # interpolation='nearest' preserves original voxel values across all resampling steps (recommended for voxel-level analysis)
    # interpolation='linear' blends values at each step (smoother edges, but compounding blur across multiple steps)
    sitk_interp  = sitk.sitkNearestNeighbor if interpolation == 'nearest' else sitk.sitkLinear
    dfield_order = 0                         if interpolation == 'nearest' else 1

    # Pre-load an arbitrary moving volume if provided, bypassing the standard path lookup
    if moving_path is not None:
        _nib = nib.load(moving_path)
        current_volume = np.asanyarray(_nib.dataobj).astype(np.float32)
        current_affine = nib.as_closest_canonical(_nib).affine
    else:
        current_volume = None
        current_affine = None

    for i in range(start_idx, end_idx):
        moving_key = PROGRESSION[i]
        paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)

        fixedimpath       = str(paths['Fixed_FilePath'])
        fixed_nib         = nib.load(fixedimpath)
        fixed_nib_canonical = nib.as_closest_canonical(fixed_nib)
        SlicerTPath       = next((p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._")), None)
        dfieldpath        = next((p for p in Path(paths['RegFold']).glob("*.pt") if not p.name.startswith("._")), None)

        warn_if_oblique(paths['Fixed_FilePath'])
        if current_volume is None:
            warn_if_oblique(paths['Moving_FilePath'])
            resampled = ApplySlicerTransform(str(paths['Moving_FilePath']), fixedimpath, str(SlicerTPath), interpolator=sitk_interp)
        else:
            resampled = ApplySlicerTransform(current_volume, fixedimpath, str(SlicerTPath), moving_affine=current_affine, interpolator=sitk_interp)

        # Canonicalize to match LandMarker's convention — field source coords index canonical space
        resampled = np.asanyarray(
            nib.as_closest_canonical(nib.Nifti1Image(resampled, fixed_nib.affine)).dataobj
        ).astype(np.float32)

        current_volume = ApplyDfield(str(dfieldpath), resampled, order=dfield_order)
        current_affine = fixed_nib_canonical.affine

    # Save result to the MovingStart stage's RegDataOut folder
    if save:
        timestamp   = datetime.now().strftime("%m%d-%H%M")
        save_paths  = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
        output_dir  = save_paths['RegDataOut']
        os.makedirs(output_dir, exist_ok=True)
        end_label = EndFixed.replace("ExVivoBlock", f"ExVivoBlock{Block:02d}").replace("BlockFace", f"Block{Block:02d}")
        output_path = os.path.join(output_dir, f"{MovingStart}RegTo{end_label}_{timestamp}.nii.gz")
        nib.save(nib.Nifti1Image(current_volume, current_affine), output_path)
        print(f"Saved to {output_path}")

    return current_volume, current_affine

def MultiStepRegDir(input_dir, RabbitID, Block, RabbitFolder, MovingStart, EndFixed, interpolation='nearest'):
    """
    Run MultiStepReg on every .nii.gz in input_dir, using each file as the moving volume.

    Outputs land in {RegDataOut}/{input_dir_name}_RegTo{end_label}/
    Output filenames: {first_two_parts_of_stem}_{regTo_suffix}_RegTo{end_label}.nii.gz
    e.g. t2map_im_MID78_interleave_MID85_regToDay3End.nii.gz
      →  t2map_im_regToDay3End_RegToBlock07.nii.gz
    """
    end_label   = EndFixed.replace("ExVivoBlock", f"ExVivoBlock{Block:02d}").replace("BlockFace", f"Block{Block:02d}")
    save_paths  = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
    out_dir     = os.path.join(save_paths['RegDataOut'], f"{os.path.basename(input_dir)}_RegTo{end_label}")
    os.makedirs(out_dir, exist_ok=True)

    vol_paths = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))
    if not vol_paths:
        print(f"No .nii.gz files found in {input_dir}")
        return

    for vol_path in vol_paths:
        stem        = os.path.basename(vol_path).replace('.nii.gz', '')
        short_stem  = '_'.join(stem.split('_')[:2])
        regt_match  = re.search(r'_[Rr]eg[Tt]o\w+', stem)
        regt_part   = regt_match.group(0) if regt_match else ''
        out_name    = f"{short_stem}{regt_part}_RegTo{end_label}.nii.gz"
        out_path    = os.path.join(out_dir, out_name)

        print(f"Processing {os.path.basename(vol_path)} → {out_name}")
        result, affine = MultiStepReg(RabbitID, Block, RabbitFolder, MovingStart, EndFixed,
                                      interpolation=interpolation, moving_path=vol_path, save=False)
        nib.save(nib.Nifti1Image(result, affine), out_path)
        print(f"  Saved → {out_path}")

#Single file run through-
#MultiStepReg(RabbitID, Block, RabbitFolder, "InVivo", "BlockFace", interpolation='nearest')

#Run through all the files in a directory-
MultiStepRegDir(RegDir, RabbitID, Block, RabbitFolder,"InVivo", "BlockFace",interpolation='nearest')

#Buggy- needs work before implementation
# resampled=compose_e_resample(SlicerTPath, dfieldpath, fixed_image, moving_image)
# output_path2 = os.path.join(paths['RegDataOut'], f"OneStepTransformApply{timestamp}.nii.gz")
# sitk.WriteImage(resampled, output_path2)

