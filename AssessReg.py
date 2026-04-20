import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import sys
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import QApplication
from ApplyTransforms import *
from RabbitPathFinder import *
from Viewer import VolumeViewer


def _norm255(arr):
    m = np.max(arr)
    return arr / m * 255 if m > 0 else arr


def dice(mask_a, mask_b):
    intersection = np.sum(mask_a & mask_b)
    total = np.sum(mask_a) + np.sum(mask_b)
    if total == 0:
        return 1.0
    return 2 * intersection / total


def hd95(mask_a, mask_b, zooms):
    """95th-percentile Hausdorff distance in mm between two binary masks."""
    surf_a = mask_a & ~binary_erosion(mask_a)
    surf_b = mask_b & ~binary_erosion(mask_b)
    coords_a = np.array(np.where(surf_a)).T * np.array(zooms)
    coords_b = np.array(np.where(surf_b)).T * np.array(zooms)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return float('nan')
    d_a2b = cKDTree(coords_b).query(coords_a)[0]
    d_b2a = cKDTree(coords_a).query(coords_b)[0]
    return float(np.percentile(np.concatenate([d_a2b, d_b2a]), 95))


## Function which takes as input a rabbit and two stages->
RabbitFolder= '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID= 'R23-055'
blockID= 5

#Provide key for moving and final fixed volume. Options- ["InVivo", "ExVivo", "ExVivoBlock", "Blockface"]
MovingStart = "InVivo"
EndFixed = "ExVivoBlock"

#Adapting this code block from Final Reg Pipeline->

#Generalizing to be able to do multiple steps->
#Provide key for moving and final fixed volume. Options- ["InVivo", "ExVivo", "ExVivoBlock", "Blockface"]
def MultiRegAssess(RabbitID, Block, RabbitFolder, MovingStart, EndFixed):
    PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]

    start_idx = PROGRESSION.index(MovingStart)
    end_idx   = PROGRESSION.index(EndFixed)

    if end_idx <= start_idx:
        print("The two volumes selected are not fit for registration")
        return

    current_volume = None
    current_affine = None

    #Each reg step here->
    for i in range(start_idx, end_idx):
        moving_key = PROGRESSION[i]
        paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)

        MovingAssessPath =  str(get_assess_data(paths['Moving_Folder']))
        FixedAssessPath = str(get_assess_data(paths['Fixed_Folder']))
        fixed_assess_nib = nib.load(FixedAssessPath)
        fixed_assess_canonical = nib.as_closest_canonical(fixed_assess_nib)

        fixedimpath       = str(paths['Fixed_FilePath'])
        fixed_nib         = nib.load(fixedimpath)
        fixed_nib_canonical = nib.as_closest_canonical(fixed_nib)
        SlicerTPath       = next((p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._")), None)
        dfieldpath        = next((p for p in Path(paths['RegFold']).glob("*.pt") if not p.name.startswith("._")), None)

        warn_if_oblique(paths['Fixed_FilePath'])
        if current_volume is None:
            warn_if_oblique(paths['Moving_FilePath'])
            resampled = ApplySlicerTransform(MovingAssessPath, fixedimpath, str(SlicerTPath))
        else:
            resampled = ApplySlicerTransform(current_volume, fixedimpath, str(SlicerTPath), moving_affine=current_affine)

        # Canonicalize to match LandMarker's convention — field source coords index canonical space
        resampled = np.asanyarray(
            nib.as_closest_canonical(nib.Nifti1Image(resampled, fixed_nib.affine)).dataobj
        ).astype(np.float32)

        current_volume = ApplyDfield(str(dfieldpath), resampled)
        current_affine = fixed_nib_canonical.affine

        fixed_assess_arr = np.asanyarray(fixed_assess_canonical.dataobj).astype(np.float32)

        # Mask both segmentations to voxels where the fixed MRI has data
        valid = fixed_nib_canonical.get_fdata() > 0
        moving_mask = (current_volume > 0.5) & valid
        fixed_mask  = (fixed_assess_arr > 0.5) & valid

        zooms = fixed_nib_canonical.header.get_zooms()[:3]
        dsc  = dice(moving_mask, fixed_mask)
        hd   = hd95(moving_mask, fixed_mask, zooms)
        label = f'{moving_key} → {PROGRESSION[i + 1]}'
        print(f'\n{label}')
        print(f'  Dice:  {dsc:.4f}')
        print(f'  HD95:  {hd:.2f} mm')

        title = label
        app = QApplication.instance() or QApplication(sys.argv)
        viewer = VolumeViewer(
            _norm255(fixed_assess_arr),
            _norm255(current_volume),
            title=title,
        )
        viewer.show()
        app.exec_()


    return current_volume, current_affine




if __name__ == '__main__':
    MultiRegAssess(RabbitID, blockID, RabbitFolder, MovingStart, EndFixed)

    #Visualize? W/ Viewer?