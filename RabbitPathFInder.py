import os
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
import numpy as np

def find_nii(directory):
    for ext in ("*.nii.gz", "*.nii"):
        matches = [p for p in Path(directory).glob(ext) if not p.name.startswith("._")]
        if matches:
            return matches[0]
    return None

def find_all_the_paths(RabbitID, Block, pathtoRabbits, Moving):
    BlockID= "Block"+f"{Block:02d}"
    RabbitFolder= os.path.join(pathtoRabbits, RabbitID)

    #Okay here's the folders to all the important stuff-
    InVivoFolder = os.path.join(RabbitFolder, 'InVivo_MR')
    ExVivoFolder = os.path.join(RabbitFolder, 'ExVivo_MR')
    ExVivo_croppedtoblock_Folder = os.path.join(ExVivoFolder, BlockID+'Reg')
    ExVivoBlockFolder = os.path.join(RabbitFolder, 'ExVivo_MRBlocked', BlockID)
    BlockFaceFolder = os.path.join(RabbitFolder, 'BlockFace_RGB', BlockID)

    # (file_folder, reg_folder) — ExVivo is the special case
    stages = {
        "InVivo": (InVivoFolder, InVivoFolder),
        "ExVivo": (ExVivoFolder, ExVivo_croppedtoblock_Folder),
        "ExVivoBlock": (ExVivoBlockFolder, ExVivoBlockFolder),
        "BlockFace": (BlockFaceFolder, BlockFaceFolder),
    }

    PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]
    moving_idx = PROGRESSION.index(Moving)
    fixed_key = PROGRESSION[moving_idx + 1]  # IndexError if Moving == "BlockFace"
    moving_file_folder, moving_reg_folder = stages[Moving]
    fixed_file_folder, _ = stages[fixed_key]

    return {
        "Moving_FilePath": find_nii(moving_file_folder),
        "Fixed_FilePath":  find_nii(fixed_file_folder),
        "Fixed_Folder":    fixed_file_folder,
        "RegFold":         os.path.join(moving_reg_folder, 'RegTransforms'),
        "RegDataOut":      os.path.join(moving_reg_folder, 'RegDataOut'),
        "RegDataProc":     os.path.join(moving_reg_folder, 'RegDataProc'),
    }

#For debugging-
def print_geometry_diagnostics(paths, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    for role, key in [("FIXED", "Fixed_FilePath"), ("MOVING", "Moving_FilePath")]:
        path = str(paths[key])
        print(f"\n--- {role}: {path} ---")

        nib_img = nib.load(path)
        print(f"  nibabel shape:      {nib_img.shape}")
        print(f"  nibabel axis codes: {nib.aff2axcodes(nib_img.affine)}")
        print(f"  nibabel affine:\n{nib_img.affine}")

        nib_can = nib.as_closest_canonical(nib_img)
        print(f"  canonical codes:    {nib.aff2axcodes(nib_can.affine)}")
        print(f"  canonical affine:\n{nib_can.affine}")

        sitk_img = sitk.ReadImage(path)
        print(f"  sitk size:      {sitk_img.GetSize()}")
        print(f"  sitk spacing:   {sitk_img.GetSpacing()}")
        print(f"  sitk origin:    {sitk_img.GetOrigin()}")
        print(f"  sitk direction:\n{np.array(sitk_img.GetDirection()).reshape(3, 3)}")

    slicer_path = next(Path(paths['RegFold']).glob("*.h5"), None)
    if slicer_path:
        print(f"\n--- SLICER TRANSFORM: {slicer_path} ---")
        t = sitk.ReadTransform(str(slicer_path))
        print(f"  type:       {t.GetName()}")
        print(f"  parameters: {np.array(t.GetParameters())}")
        print(f"  fixed params (center): {np.array(t.GetFixedParameters())}")
    else:
        print("\n  No .h5 transform found")

