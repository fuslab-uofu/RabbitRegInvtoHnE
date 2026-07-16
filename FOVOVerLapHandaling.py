#Import necessary Libraries
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

RabbitNum= "R24-082"
Block = 5

BASE_DIR = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/"
OUTPUT_DIR = "/Users/jbonaventura/Desktop/TestReg"

# Same physically-motivated bounds MultiRabAnalysis.py applies to T1/T2 maps.
T1_RANGE = (0, 3000)
T2_RANGE = (0, 200)
T2_MAX_VALID = 250


def load_mr_rename_row(rabbit_num, rename_csv_path):
    """raw_name -> standardized_name for one rabbit, restricted to standardized
    names present for every rabbit -- same 'universal features only' rule
    standardize_mr uses in MultiRabAnalysis.py."""
    wide = pd.read_csv(rename_csv_path)
    row = wide[wide['Standardized'] == rabbit_num].iloc[0]
    data = wide.drop(columns=['Standardized'])
    keep_names = set(data.columns[data.notna().all(axis=0)])
    return {raw_name: std_name for std_name, raw_name in row.items()
            if std_name != 'Standardized' and pd.notna(raw_name) and std_name in keep_names}


def find_nifti(folder, raw_name):
    matches = sorted(f for f in glob.glob(os.path.join(folder, f"{raw_name}*.nii.gz"))
                      if not os.path.basename(f).startswith('._'))
    if not matches:
        print(f"  No NIfTI found for '{raw_name}' in {folder}")
        return None
    if len(matches) > 1:
        print(f"  Multiple NIfTIs match '{raw_name}', using first: {matches}")
    return matches[0]


#Use rename script to differentiate between data collection dates and types->
Rename_Doc = "/Users/jbonaventura/Documents/MR_renames.csv"


#Load in Registered Niftis
#Example path- /System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-240/InVivo_MR/RegDataOut/Day3End_Registered_RegToBlock03
data_dir = os.path.join(BASE_DIR, RabbitNum, "InVivo_MR", "RegDataOut", f"Day3End_Registered_RegToBlock{Block:02d}")
raw_to_std = load_mr_rename_row(RabbitNum, Rename_Doc)

volumes = {}  # standardized_name -> {'data': array, 'affine':, 'header':}
for raw_name, std_name in raw_to_std.items():
    path = find_nifti(data_dir, raw_name)
    if path is None:
        continue
    img = nib.load(path)
    volumes[std_name] = {'data': img.get_fdata(), 'affine': img.affine, 'header': img.header}
    print(f"Loaded '{std_name}' <- {os.path.basename(path)}  shape={volumes[std_name]['data'].shape}")


#Zero out MaxTempProj non zero background
# Max Temp Proj has two background regions: exact 0 (voxels outside the registration
# canvas entirely) and a large flat non-zero plateau at ~body temperature (real MR
# FOV, but untreated tissue). Detect the plateau value from the data itself (mode of
# the non-zero voxels) rather than hardcoding a guessed constant, since it can drift
# slightly rabbit-to-rabbit / registration-to-registration.
maxtemp = volumes['Max Temp Proj']['data']
nonzero_vals = maxtemp[maxtemp != 0]
if nonzero_vals.size:
    uniq, counts = np.unique(np.round(nonzero_vals, 2), return_counts=True)
    background_val = uniq[np.argmax(counts)]
    print(f"Max Temp Proj: detected flat background plateau at {background_val:.2f} "
          f"({counts.max()}/{nonzero_vals.size} nonzero voxels)")
    maxtemp_clean = maxtemp.copy()
    maxtemp_clean[np.isclose(maxtemp, background_val, atol=0.1)] = 0
    volumes['Max Temp Proj']['data'] = maxtemp_clean
else:
    print("Max Temp Proj: no nonzero voxels found, skipping background cleanup")


#Apply Map thresholds
for std_name, vol in volumes.items():
    name = std_name.lower()
    if 'map' in name and 't1' in name:
        bad = (vol['data'] < T1_RANGE[0]) | (vol['data'] > T1_RANGE[1])
    elif 'map' in name and 't2' in name:
        bad = (vol['data'] < T2_RANGE[0]) | (vol['data'] > T2_MAX_VALID)
    else:
        continue
    if bad.any():
        print(f"{std_name}: {bad.sum()} voxel(s) outside valid range, zeroing")
        vol['data'][bad] = 0


#Use zero regions to mask out FOV missalignment
#Look at Day 3 as a set and Day 0 as a set then all of them bundled together
# ROI = Max Temp Proj's nonzero region after cleanup (inside the actual MR FOV) --
# that's the tissue region we actually care about losing data from.
roi = volumes['Max Temp Proj']['data'] != 0

day3_names = [n for n in volumes if 'day 3' in n.lower()]
day0_names = [n for n in volumes if 'day 0' in n.lower()]
all_names  = list(volumes.keys())  # Day 3 + Day 0 + Max Temp Proj bundled together


def report_fov_loss(names, label):
    if not names:
        print(f"{label}: no volumes in this set, skipping")
        return None
    valid = np.all([volumes[n]['data'] != 0 for n in names], axis=0)
    any_invalid_in_roi = (~valid) & roi
    print(f"{label} ({len(names)} volumes: {names}):")
    print(f"  {any_invalid_in_roi.sum()}/{roi.sum()} ROI voxels "
          f"({100 * any_invalid_in_roi.sum() / roi.sum():.1f}%) invalid in >=1 volume")
    return valid


day3_valid = report_fov_loss(day3_names, "Day 3 set")
day0_valid = report_fov_loss(day0_names, "Day 0 set")
all_valid  = report_fov_loss(all_names, "All (Day 3 + Day 0 + Max Temp Proj) bundled")


#Save Masked out Day 3-T1WCE so we can see the extent of this in Slicer
#Save to this test output folder-> /Users/jbonaventura/Desktop/TestReg
os.makedirs(OUTPUT_DIR, exist_ok=True)
t1wce_name = 'T1w CE Day 3'
if t1wce_name in volumes:
    t1wce = volumes[t1wce_name]
    for mask, mask_label in [(day3_valid, 'Day3Mask'), (day0_valid, 'Day0Mask'), (all_valid, 'BothMask')]:
        if mask is None:
            continue
        masked = t1wce['data'].copy()
        masked[~mask] = 0
        out_path = os.path.join(OUTPUT_DIR, f"{RabbitNum}_T1wCEDay3_FOVMasked_{mask_label}.nii.gz")
        nib.save(nib.Nifti1Image(masked, t1wce['affine'], t1wce['header']), out_path)
        print(f"Saved FOV-masked T1w CE Day 3 ({mask_label}) to {out_path}")

max_temp = volumes['Max Temp Proj']
out_path = os.path.join(OUTPUT_DIR, f"{RabbitNum}_MaxTempProj_BackgroundCleaned.nii.gz")
nib.save(nib.Nifti1Image(max_temp['data'], max_temp['affine'], max_temp['header']), out_path)
print(f"Saved background-cleaned Max Temp Proj to {out_path}")
