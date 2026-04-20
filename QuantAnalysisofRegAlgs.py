import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import sys
sys.path.insert(0, '/Users/jbonaventura/Documents/LandMarker/src')

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from pathlib import Path
from ApplyTransforms import *
from RabbitPathFinder import *
from NiftyTransforms import makesegfromvol, match_histograms, bin_intensities
from AssessReg import dice, hd95
from IntensityBasedRegistration import IntensityBasedRegistration
from Common.Interp import interp
import matplotlib.pyplot as plt

def load_stage_data(RabbitID, Block, RabbitFolder, MovingStart):
    """Load rigidly pre-aligned moving/fixed volumes and segmentation masks for one stage.

    Applies the Slicer rigid transform to both the moving volume and its segmentation
    so they are already in fixed voxel space — ready for intensity-based registration.

    Returns:
        moving_arr:   rigidly pre-aligned moving volume, canonical space (float32)
        fixed_arr:    fixed volume, canonical space (float32)
        moving_seg:   moving segmentation, rigidly pre-aligned, canonical space (bool)
        fixed_seg:    fixed segmentation, canonical space (bool)
        fixed_affine: affine of the canonical fixed image
        zooms:        voxel spacing of the canonical fixed image (mm)
    """
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)

    fixedimpath = str(paths['Fixed_FilePath'])
    fixed_nib = nib.load(fixedimpath)
    fixed_can = nib.as_closest_canonical(fixed_nib)
    fixed_arr = fixed_can.get_fdata().astype(np.float32)

    SlicerTPath = next(
        (p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._")), None
    )

    def _rigid_and_canonicalize(moving_path, interpolator):
        resampled = ApplySlicerTransform(moving_path, fixedimpath, str(SlicerTPath), interpolator=interpolator)
        return np.asanyarray(
            nib.as_closest_canonical(nib.Nifti1Image(resampled, fixed_nib.affine)).dataobj
        ).astype(np.float32)

    # Load in Moving and Fixed Volumes
    moving_arr = _rigid_and_canonicalize(str(paths['Moving_FilePath']), sitk.sitkLinear)

    # Load in Segmentation Masks for each
    moving_seg_path = get_assess_data(paths['Moving_Folder'])
    fixed_seg_path  = get_assess_data(paths['Fixed_Folder'])

    moving_seg = _rigid_and_canonicalize(str(moving_seg_path), sitk.sitkNearestNeighbor) > 0.5
    fixed_seg  = nib.as_closest_canonical(nib.load(str(fixed_seg_path))).get_fdata() > 0.5

    return moving_arr, fixed_arr, moving_seg, fixed_seg, fixed_can.affine, fixed_can.header.get_zooms()[:3]


def prep_for_intensity_reg(moving_arr, fixed_arr, thresh=70, maskthresh=160, kernsize=10, n_bins=8,
                           apply_mask=True, apply_binning=True):
    """Normalize, mask, and histogram-match moving and fixed volumes for intensity-based registration.

    All parameters are exposed so they can be varied as part of a parameter sweep.

    Args:
        thresh:         intensity threshold for tissue detection in makesegfromvol
        maskthresh:     filter response threshold for tissue mask in makesegfromvol
        kernsize:       box filter size for tissue mask in makesegfromvol
        n_bins:         number of intensity bins for bin_intensities
        apply_mask:     if False, skip masking and histogram matching
        apply_binning:  if False, skip intensity binning

    Returns:
        moving_prepped: preprocessed moving volume
        fixed_prepped:  preprocessed fixed volume
        mask:           binary tissue mask derived from fixed volume (None if apply_mask=False)
    """
    moving_norm = moving_arr / np.max(moving_arr) * 255 if np.max(moving_arr) > 0 else moving_arr
    fixed_norm  = fixed_arr  / np.max(fixed_arr)  * 255 if np.max(fixed_arr)  > 0 else fixed_arr

    if apply_mask:
        mask = makesegfromvol(fixed_norm, thresh=thresh, maskthresh=maskthresh, kernsize=kernsize)
        moving_out = match_histograms(moving_norm * mask, fixed_norm * mask)
        fixed_out  = fixed_norm * mask
    else:
        mask = None
        moving_out = moving_norm
        fixed_out  = fixed_norm

    if apply_binning:
        moving_out = bin_intensities(moving_out, n_bins=n_bins)
        fixed_out  = bin_intensities(fixed_out,  n_bins=n_bins)

    return moving_out, fixed_out, mask

def do_reg(moving, fixed, method='sdiff', sigma=[0.0001, 0.001, 0.1], eps=[100, 10, 1], niters=[100, 50, 4],
           lam=None, device='cpu', use_mask=True):
    """Run intensity-based registration and return the deformation field and energies.
    Args:
        moving:    pre-processed moving volume (numpy float32)
        fixed:     pre-processed fixed volume (numpy float32)
        method:    registration algorithm — 'sdiff', 'Elastic', or 'Diffeomorphic'
        sigma:     per-scale regularisation weight (list, one per scale)
        eps:       per-scale step size (list, one per scale)
        niters:    per-scale iteration count (list, one per scale)
        lam:       per-scale lambda (list, one per scale) — required for Elastic and Diffeomorphic
        device:    torch device ('cpu' or 'cuda')
        use_mask:  if True, restrict gradient to non-zero fixed voxels;
                   set False if variance equalisation produces negative values

    Returns:
        phi:      deformation field (torch tensor)
        energies: list of energy histories per scale
    """
    params = {"sigma": sigma, "eps": eps, "niters": niters}
    if lam is not None:
        params["lambda"] = lam
    phi, energies = IntensityBasedRegistration(
        fixed.astype(np.float32), moving.astype(np.float32),
        (method, params), device, use_mask=use_mask
    )
    return phi, energies


def apply_phi_to_seg(moving_seg, phi, device='cpu'):
    """Apply a LandMarker deformation field to a binary segmentation mask.

    Uses nearest-neighbor interpolation (order=0) to preserve binary values.
    """
    seg_tensor = torch.from_numpy(moving_seg.astype(np.float32)).to(device)
    warped = interp(seg_tensor, phi).cpu().numpy() > 0.5

    # phi may be spatially smaller than the input due to rescale rounding crops —
    # pad the result back to the original shape with False (background)
    if warped.shape != moving_seg.shape:
        pad = [(0, t - w) for w, t in zip(warped.shape, moving_seg.shape)]
        warped = np.pad(warped, pad, mode='constant', constant_values=0)

    return warped


# Calculate and return Dice or Hausdorff distance
def calc_difference(moving_seg, fixed_seg, zooms):
    """Compute DSC and HD95 between two binary segmentation masks.

    Both masks must already be in the same space (i.e. moving_seg has had the
    deformation field applied before being passed in here).
    """
    dsc = dice(moving_seg, fixed_seg)
    hd  = hd95(moving_seg, fixed_seg, zooms)
    return dsc, hd


#Implementation-
RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID = "R23-055"
Start = "ExVivo"
Block = 6

moving_arr, fixed_arr, moving_seg, fixed_seg, fixed_affine, fixed_zooms = load_stage_data(RabbitID, Block, RabbitFolder, Start)

moving_reg, fixed_reg, mask = prep_for_intensity_reg(moving_arr, fixed_arr, thresh=70, maskthresh=160, kernsize=10, n_bins=8,
                           apply_mask=True, apply_binning=True)

print(moving_reg.shape, fixed_reg.shape)

results = []
vlist= [0.01,0.005, 0.001,0.0005,0.0001]
for q in range(5):
    v =vlist[q]
    print(v)
    phi, energies = do_reg(moving_reg, fixed_reg, method='sdiff', sigma=[v, 0.001, 0.1], eps=[100, 10, 1], niters=[100, 50, 4], lam=None, device='cpu', use_mask=True)

    if any(np.isnan(e) for scale in energies for e in scale):
        print(f"sigma={v} diverged (NaN energies) — skipping")
        continue

    warped_seg = apply_phi_to_seg(moving_seg, phi)
    dsc, hd = calc_difference(warped_seg, fixed_seg, fixed_zooms)

    results.append({
        'var': v,
        'dsc': dsc,
        'hd95': hd,
        'energies': energies,  # [[l2...], [reg...], [total...]]
    })

#breakpoint()  # drop into debugger here to explore results before plotting

var = [r['var'] for r in results]
dscs = [r['dsc'] for r in results]
hd95 = [r['hd95'] for r in results]
final_energy = [r['energies'][2][-1] for r in results]


plt.plot(var, dscs)
plt.plot( var, hd95)
plt.plot(var, final_energy)
plt.show(block=True)

        # Compare at the end in plots!!
