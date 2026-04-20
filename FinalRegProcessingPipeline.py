#Import libraries-
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import numpy as np
import nibabel as nib
from ApplyTransforms import *
from RabbitPathFinder import *
from datetime import datetime
import matplotlib.pyplot as plt
import glob

#Set which Rabbit and Block we want, RabbitData is where it all lives, folder structure matters here->
RabbitFolder='/Users/jbonaventura/Downloads/RabbitData'
RabbitID="R23-055"
Block = 6

#Generalizing to be able to do multiple steps->
#Provide key for moving and final fixed volume. Options- ["InVivo", "ExVivo", "ExVivoBlock", "Blockface"]
def MultiStepReg(RabbitID, Block, RabbitFolder, MovingStart, EndFixed):
    PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]

    start_idx = PROGRESSION.index(MovingStart)
    end_idx   = PROGRESSION.index(EndFixed)

    if end_idx <= start_idx:
        print("The two volumes selected are not fit for registration")
        return

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
            resampled = ApplySlicerTransform(str(paths['Moving_FilePath']), fixedimpath, str(SlicerTPath))
        else:
            resampled = ApplySlicerTransform(current_volume, fixedimpath, str(SlicerTPath), moving_affine=current_affine)

        # Canonicalize to match LandMarker's convention — field source coords index canonical space
        resampled = np.asanyarray(
            nib.as_closest_canonical(nib.Nifti1Image(resampled, fixed_nib.affine)).dataobj
        ).astype(np.float32)

        current_volume = ApplyDfield(str(dfieldpath), resampled)
        current_affine = fixed_nib_canonical.affine

    # Save result to the MovingStart stage's RegDataOut folder
    timestamp   = datetime.now().strftime("%m%d-%H%M")
    save_paths  = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
    output_dir  = save_paths['RegDataOut']
    os.makedirs(output_dir, exist_ok=True)
    end_label = EndFixed.replace("ExVivoBlock", f"ExVivoBlock{Block:02d}").replace("BlockFace", f"Block{Block:02d}")
    output_path = os.path.join(output_dir, f"{MovingStart}RegTo{end_label}_{timestamp}.nii.gz")
    nib.save(nib.Nifti1Image(current_volume, current_affine), output_path)
    print(f"Saved to {output_path}")

    return current_volume, current_affine

MultiStepReg("R23-055", 6, '/Users/jbonaventura/Downloads/RabbitData', "InVivo", "BlockFace")

#Buggy- needs work before implementation
# resampled=compose_e_resample(SlicerTPath, dfieldpath, fixed_image, moving_image)
# output_path2 = os.path.join(paths['RegDataOut'], f"OneStepTransformApply{timestamp}.nii.gz")
# sitk.WriteImage(resampled, output_path2)

