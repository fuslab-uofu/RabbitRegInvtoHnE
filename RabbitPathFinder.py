import os
import csv
import glob
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
import numpy as np

# The same ceph share is mounted at a different path on each machine: the Mac, then mara.
CEPH_HIFU_MOUNTS = ['/System/Volumes/Data/ceph/hifu', '/ceph/hifu']
RABBIT_REG_PROJ = 'users/jbonaventura/RabbitRegistrationProj'


def ceph_hifu_base():
    """This machine's ceph hifu mount: the first of CEPH_HIFU_MOUNTS that exists."""
    for mount in CEPH_HIFU_MOUNTS:
        if os.path.isdir(mount):
            return mount
    raise FileNotFoundError(f"No ceph hifu mount found here, tried {CEPH_HIFU_MOUNTS}")


def rabbit_reg_proj_dir():
    """RabbitRegistrationProj on this machine's ceph mount."""
    return os.path.join(ceph_hifu_base(), RABBIT_REG_PROJ)


def rabbit_data_dir():
    """RabbitData (one R<id> folder per rabbit) on this machine's ceph mount."""
    return os.path.join(rabbit_reg_proj_dir(), 'RabbitData')


def prediction_results_dir():
    """RadPathPredictions (MR -> H&E model outputs) on this machine's ceph mount."""
    return os.path.join(rabbit_reg_proj_dir(), 'RadPathPredictions')


def he_projection_dir(label_source):
    """HnE_Proj_toMRI/<label source>/ on this machine's ceph mount: H&E class maps on the MR grid
    combined over a rabbit's blocks (ProjectClassPredictionsToMR.py), for viewing."""
    return os.path.join(rabbit_reg_proj_dir(), 'HnE_Proj_toMRI', label_source)


def prediction_run_dir(target_source, mr_norm_dir):
    """RadPathPredictions/<target source>_<MR normalization>/: every MR -> H&E model output
    predicting one H&E target set (e.g. the 'R3_softmax' annotations) from one MR
    normalization, with a subfolder per stage."""
    return os.path.join(prediction_results_dir(), f'{target_source}_{mr_norm_dir}')


def load_ready_rabbit_blocks(csv_path=None):
    """[(rabbit ID without its 'R' prefix, block number), ...] from ReadyRabbits.csv (default:
    the one in RabbitRegistrationProj), the rabbit blocks whose MR registration and histology
    are both ready for analysis. One pair per row: a rabbit with several analysed blocks has
    one row per block."""
    if csv_path is None:
        csv_path = os.path.join(rabbit_reg_proj_dir(), 'ReadyRabbits.csv')
    with open(csv_path, newline='') as f:
        return [(row['rabbit'].strip().removeprefix('R'), int(row['block']))
                for row in csv.DictReader(f)]


def block_label(rabbit_id, block_num):
    """'R23-055 Block05': names one analysed rabbit block in printouts and plot titles."""
    return f'R{rabbit_id} Block{block_num:02d}'


def block_analysis_dir(base_dir, rabbit_id, block_num):
    """Per-block Analysis folder that holds that block's H&E volume and feature CSVs."""
    return os.path.join(base_dir, f'R{rabbit_id}', 'Analysis', f'Block{block_num:02d}')


def day3end_mr_dir(base_dir, rabbit_id, mr_norm_dir):
    """Day3End_Registered MR folder for one rabbit: its mr_norm_dir subfolder, or the raw
    volumes themselves when mr_norm_dir is ''."""
    day3end_dir = os.path.join(base_dir, f'R{rabbit_id}', 'InVivo_MR/InVMRDataSets/Day3End_Registered')
    return os.path.join(day3end_dir, mr_norm_dir) if mr_norm_dir else day3end_dir


def class_predictions_dir(rabbit_id, block_num, label_source):
    """ProjectClassPredictionsToMR.py's output folder for one block and label source."""
    return (Path(__file__).parent / 'TestData' /
            f'R{rabbit_id}_Block{block_num:02d}_ClassPredictions_{label_source}')


def class_fraction_col(class_name):
    """Feature-table column holding a voxel's (or tile's / patch's mean) fraction of class_name."""
    return f'{class_name}_fraction'


def mr_norm_csv_suffix(mr_norm_dir):
    """Feature-CSV filename suffix naming the Day3End_Registered subfolder the MR volumes
    were read from ('' = the raw volumes), so each normalization keeps its own CSVs."""
    return f'_{mr_norm_dir}' if mr_norm_dir else '_raw'


def find_nii(directory):
    for ext in ("*.nii.gz", "*.nii"):
        matches = [p for p in Path(directory).glob(ext) if not p.name.startswith("._")]
        if matches:
            return matches[0]
    return None

def find_single_transform(reg_fold, ext):
    """Return the one transform file with extension ext ('.h5' or '.pt') in reg_fold.
    Stops if there are none or several -- with several, there's no reliable way to tell
    which is current, and picking the wrong one silently gives a stale registration."""
    matches = sorted(p for p in Path(reg_fold).glob(f"*{ext}") if not p.name.startswith("._"))
    if not matches:
        raise FileNotFoundError(f"No {ext} transform found in {reg_fold}")
    if len(matches) > 1:
        names = "\n  ".join(p.name for p in matches)
        raise RuntimeError(
            f"Found {len(matches)} {ext} transforms in {reg_fold}:\n  {names}\n"
            f"Delete the old one(s) so only the current transform is left, then rerun."
        )
    return matches[0]

def get_assess_data(parent_folder):
    assess_folder = os.path.join(parent_folder, 'RegAssessData')
    nifti_path = find_nii(assess_folder)
    #Can add LMs as well here when we're ready
    return nifti_path



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
        "Moving_Folder":   moving_file_folder,
        "Fixed_FilePath":  find_nii(fixed_file_folder),
        "Fixed_Folder":    fixed_file_folder,
        "RegFold":         os.path.join(moving_reg_folder, 'RegTransforms'),
        "RegDataOut":      os.path.join(moving_reg_folder, 'RegDataOut'),
        "RegDataProc":     os.path.join(moving_reg_folder, 'RegDataProc'),
    }

def find_hne_slide_path(RabbitID, Block, pathtoRabbits, czi_path, normalized=True):
    """Resolve the H&E slide to actually read: the normalized TIFF written by
    TilingCorrection.py (default), or the original raw CZI (normalized=False)."""
    if not normalized:
        return czi_path
    BlockID = "Block" + f"{Block:02d}"
    RabbitFolder = os.path.join(pathtoRabbits, RabbitID)
    czi_stem = os.path.splitext(os.path.basename(czi_path))[0]
    return os.path.join(RabbitFolder, 'HnE', BlockID, 'NormalizedFull',
                         czi_stem + '_corrected_fullres.tif')


def find_day0_paths(RabbitID, pathtoRabbits):
    day0_dir = os.path.join(pathtoRabbits, RabbitID, 'InVivo_MR', 'InVMRDataSets', 'Day0')
    return {
        'day0_dir': day0_dir,
        'vols':     sorted(glob.glob(os.path.join(day0_dir, '*.nii.gz'))),
    }


def find_day3_paths(RabbitID, pathtoRabbits):
    day3_dir = os.path.join(pathtoRabbits, RabbitID, 'InVivo_MR', 'InVMRDataSets', 'Day3')

    log_matches = glob.glob(os.path.join(day3_dir, '*InVDay3Log.csv'))
    if not log_matches:
        raise FileNotFoundError(f"No InVDay3Log.csv found in {day3_dir}")

    stem_to_category = {}
    with open(log_matches[0]) as f:
        for row in csv.DictReader(f):
            stem_to_category[row['File Name'].strip()] = row['Home'].strip()

    start_vols, end_vols, fixed_vols = [], [], []
    for path in sorted(glob.glob(os.path.join(day3_dir, '*.nii.gz'))):
        stem = os.path.basename(path).replace('.nii.gz', '')
        category = stem_to_category.get(stem)
        if category == 'Start':     start_vols.append(path)
        elif category == 'End':     end_vols.append(path)
        elif category == 'Fixed':   fixed_vols.append(path)

    day3end_registered_dir = os.path.join(os.path.dirname(day3_dir), 'Day3End_Registered')

    return {
        'day3_dir':            day3_dir,
        'start_vols':          start_vols,
        'end_vols':            end_vols,
        'fixed_vols':          fixed_vols,
        'transform_path':      os.path.join(day3_dir, 'Day3_end_to_start_Transform.h5'),
        'inv_transform_cache': os.path.join(day3_dir, 'Day3_end_to_start_inv_cached.h5'),
        'aps_transform_folder': os.path.join(os.path.dirname(day3_dir), 'Transforms'),
        'day3end_registered_dir': day3end_registered_dir,
        # Day3 start/fixed volumes already resampled onto the End-Day3 (target_space="InVivo") grid
        'registered_vols':     sorted(glob.glob(os.path.join(day3end_registered_dir, '*_regToDay3End.nii.gz'))),
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

