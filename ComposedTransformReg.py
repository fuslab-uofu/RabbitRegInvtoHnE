#Import libraries-
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates
from RabbitPathFinder import *
from ApplyTransforms import warn_if_oblique

# Mirrors FinalRegProcessingPipeline.py's MultiStepReg, but composes all stages
# between MovingStart and EndFixed into ONE combined field first and resamples once,
# instead of resampling once per stage.


PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]


#Loads and saves temp .h5 file to circumvent HDF5 file locking when loading from Ceph.
def _load_transform(path):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        return sitk.ReadTransform(tmp_path)
    finally:
        os.unlink(tmp_path)

def _load_stage(RabbitID, Block, RabbitFolder, moving_key, debug=False):
    #Everything needed for this stage's fixed->moving point pull:
    # bare-affine M/t/c (Matrix, translation, center) .pt deformation field cropped to fixed cannonical grid shape- to account for dimsdivbyfour issues.
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)
    warn_if_oblique(paths['Fixed_FilePath'])
    fixed_nib_canonical = nib.as_closest_canonical(nib.load(str(paths['Fixed_FilePath'])))

    slicer_path = next(p for p in Path(paths['RegFold']).glob('*.h5') if not p.name.startswith('._'))
    dfield_path = next(p for p in Path(paths['RegFold']).glob('*.pt') if not p.name.startswith('._'))

    transform = _load_transform(slicer_path)
    assert transform.GetName() in ('AffineTransform', 'MatrixOffsetTransformBase'), (
        f"{moving_key} stage's .h5 is a {transform.GetName()}, not a bare affine — "
        f"this composer only handles affine+deformation-field stages."
    )
    params = np.array(transform.GetParameters())
    M = params[:9].reshape(3, 3)
    t = params[9:]
    c = np.array(transform.GetFixedParameters()[:3])

    fixed_key = PROGRESSION[PROGRESSION.index(moving_key) + 1]
    label = f"{moving_key}->{fixed_key}"

    dfield_raw = torch.load(str(dfield_path), map_location='cpu')['data'].numpy()  # (3, X, Y, Z), 1-indexed
    X, Y, Z = fixed_nib_canonical.shape[:3]
    dfield = dfield_raw[:, :X, :Y, :Z]

    if debug:
        raw_shape = dfield_raw.shape[1:]
        match = "match" if raw_shape == (X, Y, Z) else "MISMATCH"
        print(f"  [{label}] .pt raw field shape {raw_shape} vs fixed canonical shape "
              f"{(X, Y, Z)} -- {match}")

    return {'fixed_nib': fixed_nib_canonical, 'M': M, 't': t, 'c': c, 'dfield': dfield, 'label': label}


def _stage_pull_points(points_ras_fixed, stage, debug=False):
    """Vectorized fixed-physical -> moving-physical point pull for one stage: the
    .pt field operates first (within the fixed grid), then the affine carries the
    result into the moving image's physical space — same order MultiStepReg applies
    ApplyDfield after ApplySlicerTransform, just point-wise instead of image-wise."""
    fixed_nib = stage['fixed_nib']
    #physical coordinates of fixed space are converted to voxel index space->
    hom = np.column_stack([points_ras_fixed, np.ones(len(points_ras_fixed))])
    vox_fixed = (np.linalg.inv(fixed_nib.affine) @ hom.T).T[:, :3]

    dfield = stage['dfield']

    # A point that falls outside the field's own array here are flagged as invalid instead of clamped.
    shape = dfield.shape[1:]  # Dfield (X, Y, Z) extent.
    #Check for out of bounds voxels, less than zero or bigger than the extent of the dfield, boolean oob mask->
    oob = (
        (vox_fixed[:, 0] < 0) | (vox_fixed[:, 0] > shape[0] - 1) |
        (vox_fixed[:, 1] < 0) | (vox_fixed[:, 1] > shape[1] - 1) |
        (vox_fixed[:, 2] < 0) | (vox_fixed[:, 2] > shape[2] - 1)
    )

    # Reports how many points fell outside the field and their voxel-index range,
    # for diagnosing the clamping/wedge-artifact issue this OOB check guards against.
    if debug:
        n_oob = int(oob.sum())
        print(f"  [{stage.get('label', '?')}] field shape {shape}: "
              f"{n_oob:,}/{len(vox_fixed):,} points ({100 * n_oob / max(len(vox_fixed), 1):.2f}%) "
              f"land outside the field before clamping")
        if n_oob:
            oob_pts = vox_fixed[oob]
            print(f"    out-of-bounds voxel-index range: "
                  f"x[{oob_pts[:, 0].min():.1f},{oob_pts[:, 0].max():.1f}] "
                  f"y[{oob_pts[:, 1].min():.1f},{oob_pts[:, 1].max():.1f}] "
                  f"z[{oob_pts[:, 2].min():.1f},{oob_pts[:, 2].max():.1f}]")

    # 1-indexed Landmarker-style field convention:
    # channels [2]=X,[1]=Y,[0]=Z, subtract 1 for 0-indexed output. dfield's own array axes
    # are (X,Y,Z), same order as vox_fixed's columns — no axis reversal before sampling.
    #Dfield applied to fixed voxel coordinates->
    coords = vox_fixed.T  # (3, N) x,y,z for map_coordinates
    fx = map_coordinates(dfield[2], coords, order=1, mode='nearest') - 1
    fy = map_coordinates(dfield[1], coords, order=1, mode='nearest') - 1
    fz = map_coordinates(dfield[0], coords, order=1, mode='nearest') - 1
    vox_intermediate = np.column_stack([fx, fy, fz])

    #Updated Voxel coordinates are then converted to the fixed nifti affine physical coordinate space->
    hom2 = np.column_stack([vox_intermediate, np.ones(len(vox_intermediate))])
    point_ras_intermediate = (fixed_nib.affine @ hom2.T).T[:, :3]
    #Converted to lps space- Slicer transform standard.
    point_lps_intermediate = point_ras_intermediate * np.array([-1., -1., 1.])

    # Matrix, translation and center of affine transform gathered->
    M, t, c = stage['M'], stage['t'], stage['c']
    #points brought to origin, Multiplied by affine matrix, shift back to center and then translated>
    point_lps_moving = (point_lps_intermediate - c) @ M.T + c + t
    return point_lps_moving * np.array([-1., -1., 1.]), oob


def _pull_points_through_chain(points_ras_end, stages, debug=False):
    """stages: [(InVivo->ExVivo), (ExVivo->ExVivoBlock), (ExVivoBlock->BlockFace), ...]
    in forward (PROGRESSION) order. Walked in reverse to pull EndFixed-space points
    back to MovingStart-space, one stage at a time.

    A point flagged out-of-bounds at any stage stays invalid for the rest of the
    chain even if a later stage's clamp happens to land back in-bounds — the clamped
    position from the failing stage was already garbage, so nothing downstream of it
    is trustworthy either."""
    p = points_ras_end
    invalid = np.zeros(len(points_ras_end), dtype=bool)
    #For each stage pull the points through and then feed them into the next stage-
    for stage in reversed(stages):
        p, oob = _stage_pull_points(p, stage, debug=debug)
        invalid |= oob
    return p, invalid


def ComposedReg(RabbitID, Block, RabbitFolder, MovingStart, EndFixed, interpolation='nearest',
                 moving_path=None, save=True, debug=False, invalid_fill_value=0,
                 save_moving_vox_path=None):
    #Convert Moving and Fixed keys to Indicies:
    start_idx = PROGRESSION.index(MovingStart)
    end_idx = PROGRESSION.index(EndFixed)
    if end_idx <= start_idx:
        print("The two volumes selected are not fit for registration")
        return

    order = 0 if interpolation == 'nearest' else 1

    print(f"Loading {end_idx - start_idx} stage(s) ({', '.join(PROGRESSION[start_idx:end_idx])})...")
    #returns a list of dicts, one for each reg stage with fixed volume geometry, M,t,c of affine transform, and dfield.
    stages = [_load_stage(RabbitID, Block, RabbitFolder, PROGRESSION[i], debug=debug) for i in range(start_idx, end_idx)]

    #Load in moving volume, either through explicit path or from default location in RabbitData Folder->
    if moving_path is not None:
        moving_nib_canonical = nib.as_closest_canonical(nib.load(moving_path))
    else:
        moving_paths = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
        warn_if_oblique(moving_paths['Moving_FilePath'])
        moving_nib_canonical = nib.as_closest_canonical(nib.load(str(moving_paths['Moving_FilePath'])))
    moving_arr = np.asanyarray(moving_nib_canonical.dataobj).astype(np.float32)
    moving_affine = moving_nib_canonical.affine

    #Grab geometry of end fixed volume shape->
    end_nib = stages[-1]['fixed_nib']
    X, Y, Z = end_nib.shape[:3]
    N = X * Y * Z
    print(f"Composing into one field: {X}x{Y}x{Z} = {N:,} voxels...")

    # Make integer voxel index arrays spanning the end fixed volume geometry->
    xi, yi, zi = np.mgrid[0:X, 0:Y, 0:Z]
    #Reshape into an (N,3) array- one row per voxel with its (x,y,z) index
    end_vox = np.column_stack([xi.ravel(), yi.ravel(), zi.ravel()]).astype(np.float64)
    #Column of ones added, Nx4 for affine multiplication-
    hom = np.column_stack([end_vox, np.ones(N)])
    #Multiplied by end fixed volume affine resulting in Nx3 array of physical RAS coords for every voxel->
    end_ras = (end_nib.affine @ hom.T).T[:, :3]


    #Points pulled through all stages resulting in an Nx3 array of physical RAS coords in the original moving volume-
    moving_ras, invalid = _pull_points_through_chain(end_ras, stages, debug=debug)
    print(f"{invalid.sum():,}/{N:,} voxels ({100 * invalid.mean():.2f}%) fell outside a stage's "
          f"field at some point in the chain — filling with {invalid_fill_value}")

    #physical RAS coors in original volume converted to indicies->
    hom2 = np.column_stack([moving_ras, np.ones(N)])
    moving_vox = (np.linalg.inv(moving_affine) @ hom2.T).T[:, :3]


    #Output for assessing discontinuities in the field- not used for anything else-
    if save_moving_vox_path is not None:
        # Continuous InVivo-voxel-index coordinates laid out on the BlockFace output grid
        # so this lines up spatially with the rest of BlockFace-space in Slicer.
        # Saved as 3 separate scalar volumes: Slicer doesn't expose clean per-channel
        # viewing for a combined multi-component NIfTI.
        p = Path(save_moving_vox_path)
        suffix = ''.join(p.suffixes)
        base = str(p)[:-len(suffix)] if suffix else str(p)
        for idx, axis in enumerate(['i', 'j', 'k']):
            axis_vol = moving_vox[:, idx].reshape(X, Y, Z).astype(np.float32)
            axis_path = f"{base}_{axis}{suffix}"
            nib.save(nib.Nifti1Image(axis_vol, end_nib.affine), axis_path)
            print(f"Saved moving_vox {axis} component to {axis_path}")

    print("Resampling in one pass...")
    coords = moving_vox.T  # (3, N) — matches moving_arr's own (i,j,k) array-axis order
    #Resamples moving_arr at each coord, returning N intensity values which can then be reshaped into fixed array shape.
    result = map_coordinates(moving_arr, coords, order=order, mode='constant', cval=0).reshape(X, Y, Z)
    result[invalid.reshape(X, Y, Z)] = invalid_fill_value

    if save:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        save_paths = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
        output_dir = save_paths['RegDataOut']
        os.makedirs(output_dir, exist_ok=True)
        end_label = EndFixed.replace("ExVivoBlock", f"ExVivoBlock{Block:02d}").replace("BlockFace", f"Block{Block:02d}")
        output_path = os.path.join(output_dir, f"{MovingStart}ComposedRegTo{end_label}_{timestamp}.nii.gz")
        nib.save(nib.Nifti1Image(result, end_nib.affine), output_path)
        print(f"Saved to {output_path}")

    return result, end_nib.affine


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID = "R24-240"
    Block = 3

    ComposedReg(RabbitID, Block, RabbitFolder, 'InVivo', 'BlockFace', interpolation='linear')
