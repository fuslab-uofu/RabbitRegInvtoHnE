import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd


def load_mr_volumes(mr_dir):
    """Load every MR contrast volume in mr_dir, keyed by filename stem (minus
    _regToDay3End). These share the pipeline's common MR grid directly (see
    ResampleToDay3End.py's INVIVO_REF_PATH fix) -- no axis correction needed."""
    volumes = {}
    for f in sorted(glob.glob(os.path.join(mr_dir, '*.nii.gz'))):
        stem = os.path.basename(f).replace('_regToDay3End.nii.gz', '')
        volumes[stem] = nib.load(f).get_fdata(dtype=np.float32)
    return volumes


def mr_grid_reference(mr_dir):
    """One MR volume's NIfTI from mr_dir, for its grid (shape + affine): every contrast
    load_mr_volumes reads from there shares it."""
    return nib.load(sorted(glob.glob(os.path.join(mr_dir, '*.nii.gz')))[0])


def extract_voxelwise_features(target_volumes, mask_volumes, extra_volumes=None):
    """Mask to voxels with real (non-NaN) H&E data in every target_volumes map
    ({column name: volume}) AND real (nonzero -- MR's own no-data convention)
    data in every contrast in mask_volumes, then read the voxel's location,
    every target value, every mask_volumes contrast, and every extra_volumes
    feature at exactly those same array positions.

    mask_volumes vs extra_volumes: mask_volumes drives the validity mask (0
    must genuinely mean "no data" for every volume passed here -- true for raw
    MR contrasts, confirmed empirically). extra_volumes are derived feature
    maps (gradient, Frangi, neighborhood stats, ...) included as output
    columns but never checked against the mask -- a derived value can
    legitimately be exactly 0 (e.g. gradient in a locally flat patch of real
    tissue) without meaning "no data", so it must never be allowed to drop an
    otherwise-valid voxel the way a raw MR contrast is allowed to.

    target_volumes/mask_volumes/extra_volumes must all already share one grid (see
    load_mr_volumes). vx/vy/vz are the voxel's own (axis0, axis2, axis1)
    indices on that shared grid -- plain location, not a cross-registration
    stable ID, since every volume here already lives in the same coordinate
    system (no more voxel-ID label volume in this pipeline). Staying in
    volume space end to end (rather than converting back to an id-keyed point
    list) means duplicate-fragment aggregation is already resolved by the
    painting step. Takes already-loaded arrays rather than paths so a caller
    extracting multiple things from the same rabbit's data (e.g. this plus a
    tile-wise extraction) only loads each volume once."""
    extra_volumes = extra_volumes or {}
    mask = np.logical_and.reduce([np.isfinite(arr) for arr in target_volumes.values()])
    for arr in mask_volumes.values():
        mask &= (arr != 0)
    print(f"{int(mask.sum()):,} voxel(s) with valid H&E and every MR contrast")

    vx, vz, vy = np.nonzero(mask)
    result = {'vx': vx, 'vy': vy, 'vz': vz}
    for name, arr in target_volumes.items():
        result[name] = arr[mask]
    for name, arr in mask_volumes.items():
        result[name] = arr[mask]
    for name, arr in extra_volumes.items():
        result[name] = arr[mask]
    return pd.DataFrame(result)


if __name__ == '__main__':
    Rabbit = 'R24-240'
    block_no = 3
    Block = f'Block{block_no:02d}'
    CEPH_BASE = '/System/Volumes/Data/ceph/hifu'
    root_dir = os.path.join(CEPH_BASE, 'users/jbonaventura/RabbitRegistrationProj/RabbitData')
    rabbase = os.path.join(root_dir, Rabbit)

    mr_dir = os.path.join(rabbase, 'InVivo_MR/InVMRDataSets/Day3End_Registered/Normalized')

    analysis_dir = os.path.join(rabbase, 'Analysis', Block)
    FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
    # Produced by BuildInVivoFeatureBatch.py -- run that first for this rabbit/block.
    he_volume_path = os.path.join(analysis_dir, f'{Rabbit}_{Block}_RL_iter1.nii.gz')

    mr_volumes = load_mr_volumes(mr_dir)
    print(f"Loaded {len(mr_volumes)} MR contrast(s): {list(mr_volumes.keys())}")

    he_arr = nib.load(he_volume_path).get_fdata(dtype=np.float32)
    merged = extract_voxelwise_features({FEATURE_COL: he_arr}, mr_volumes)

    out_path = os.path.join(analysis_dir, 'voxel_features_by_location.csv')
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows -> {out_path}")
