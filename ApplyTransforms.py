import h5py
import numpy as np
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

# Applying Deformation Field-
def ApplyDfield(dfpath, movingvol):
    # Load the warp field
    data = torch.load(dfpath, map_location='cpu')
    field = data['data'].numpy()
    # print(f"field[0] range: {field[0].min():.1f} – {field[0].max():.1f}")
    # print(f"field[1] range: {field[1].min():.1f} – {field[1].max():.1f}")
    # print(f"field[2] range: {field[2].min():.1f} – {field[2].max():.1f}")
    # print(f"movingvol shape (X,Y,Z): {movingvol.shape}")
    #
    # print(f"field shape: {field.shape}, movingvol shape: {movingvol.shape}")
    # Crop field to movingvol shape (field may have been generated on zero-padded volume)
    X, Y, Z = movingvol.shape
    field = field[:, :X, :Y, :Z]
    # Reorder channels from (Z, Y, X) to (X, Y, Z) for map_coordinates and subtract 1 to convert from 1-indexed to 0-indexed
    coords = [field[2] - 1, field[1] - 1, field[0] - 1]
    # Resample
    registered = map_coordinates(movingvol, coords, order=1, mode='constant', cval=0)
    return(registered)

def warn_if_oblique(nifti_path):
    """Print a warning with the affine if the image has an oblique (non-axis-aligned) orientation.
    Oblique affines in this pipeline are unexpected and may indicate the wrong file is being used."""
    img = nib.load(str(nifti_path))
    affine = img.affine
    norms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = affine[:3, :3] / norms
    off_diagonal = direction - np.diag(np.diag(direction))
    if np.max(np.abs(off_diagonal)) > 0.01:
        print(f"WARNING: oblique affine detected in {nifti_path}")
        print(f"  axis codes: {nib.aff2axcodes(affine)}")
        print(f"  affine:\n{affine}")

def nib_to_sitk(array, nib_affine):
    """Convert a nibabel-convention (X,Y,Z) numpy array + RAS affine to a SimpleITK image.
    Handles the RAS (nibabel) -> LPS (SimpleITK) coordinate system conversion."""
    # Extract spacing from column norms of the affine's 3x3 rotation-scale block
    spacing = np.sqrt((nib_affine[:3, :3] ** 2).sum(axis=0))
    # Normalized per-axis direction vectors in RAS (columns of the 3x3 block)
    ras_direction = nib_affine[:3, :3] / spacing
    # RAS -> LPS: negate the X and Y rows
    lps_flip = np.diag([-1., -1., 1.])
    lps_direction = lps_flip @ ras_direction
    lps_origin    = lps_flip @ nib_affine[:3, 3]

    # SimpleITK expects (Z,Y,X) array ordering; nibabel uses (X,Y,Z)
    sitk_array = np.transpose(array, (2, 1, 0)).astype(np.float32)
    img = sitk.GetImageFromArray(sitk_array)
    img.SetSpacing(spacing.tolist())
    img.SetDirection(lps_direction.flatten().tolist())
    img.SetOrigin(lps_origin.tolist())
    return img

def ApplySlicerTransform(moving, fixedimpath, SlicerTPath,  moving_affine=None):
    fixed_image = sitk.ReadImage(fixedimpath)
    transform = sitk.ReadTransform(SlicerTPath)
    #Case #1- first reg step-
    if isinstance(moving, (str, Path)):
        #Load in for sitk-
        moving_image = sitk.ReadImage(moving)
    else:
    #Case #2- moving volume from previous step-
        moving_image = nib_to_sitk(moving, moving_affine)
    # Resample the moving image into the fixed image space
    resampled_sitk = sitk.Resample(
        moving_image,  # image to resample
        fixed_image,  # reference grid (defines output space)
        transform,  # the transform
        sitk.sitkLinear,  # interpolation method
        0.0,  # default pixel value for voxels outside the moving image
        moving_image.GetPixelID()
    )

    #Save if desired-
    #sitk.WriteImage(resampled_sitk, '/Users/jbonaventura/Desktop/OutputTest33/TestOut32.nii.gz')
    # Convert to numpy: SimpleITK is (Z,Y,X), transpose to (X,Y,Z) for map_coordinates
    resampled_np = sitk.GetArrayFromImage(resampled_sitk)
    resampled_np = np.transpose(resampled_np, (2, 1, 0)).astype(np.float32)
    return(resampled_np)
#
# #Image and Transform Paths-
# movingimpath='/Users/jbonaventura/Downloads/RabbitData/R23-055/ExVivo_MRBlocked/Block06/ExVMR_Block06Cropped.nii'
# fixedimpath = '/Users/jbonaventura/Downloads/RabbitData/R23-055/BlockFace_RGB/Block06/Grey_UnMasked_DownSampled.nii.gz'
# # fixed_image = sitk.ReadImage(fixedimpath)
# # moving_image = sitk.ReadImage(movingimpath)
# SlicerTPath = '/Users/jbonaventura/Downloads/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegTransforms/Block06toRGB.h5'
# dfieldpath = '/Users/jbonaventura/Downloads/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegTransforms/LMExVtoBlockFace.pt'
#
# fixed_nib = nib.load(fixedimpath)
# fixed_nib_canonical = nib.as_closest_canonical(fixed_nib)
#
# resampled_SlicerT = ApplySlicerTransform(movingimpath, fixedimpath, SlicerTPath)
# # Canonicalize movingvol to match LandMarker's convention — field source coords index canonical space
# resampled_SlicerT = np.asanyarray(
#     nib.as_closest_canonical(nib.Nifti1Image(resampled_SlicerT, fixed_nib.affine)).dataobj
# ).astype(np.float32)
#
# registered = ApplyDfield(dfieldpath, resampled_SlicerT)
# print(np.max(registered), np.min(registered))
#
# # Save with canonical affine to match LandMarker's coordinate convention
# out = nib.Nifti1Image(registered, fixed_nib_canonical.affine, fixed_nib_canonical.header)
# nib.save(out, '/Users/jbonaventura/Desktop/OutputTest33/TestOut33.nii.gz')

def vox_to_phys(image, vox):
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    origin = np.array(image.GetOrigin())
    return origin + (direction @ (spacing[:, np.newaxis] * vox.T)).T

#Approach number2- combining transforms to reduce resampling-
def pt_dfield_to_sitk(dfpath, fixed_image, moving_image):
    data = torch.load(dfpath, map_location='cpu')
    field = data['data'].numpy()  # [3, X, Y, Z], 1-indexed moving voxel coords

    # Crop field to fixed image size (field may have been generated on zero-padded volume)
    fw, fh, fd = fixed_image.GetSize()  # SimpleITK returns (X, Y, Z)
    field = field[:, :fw, :fh, :fd]

    X, Y, Z = field.shape[1:]
    

    # Identity grid — 0-indexed voxel coordinates in fixed space
    gx, gy, gz = np.mgrid[0:X, 0:Y, 0:Z]
    fixed_vox = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float64)

    # Moving voxel coords from field (0-indexed), reordered to match map_coordinates convention
    moving_vox = np.stack([field[2] - 1, field[1] - 1, field[0] - 1], axis=-1).reshape(-1, 3).astype(np.float64)

    fixed_phys = vox_to_phys(fixed_image, fixed_vox)
    moving_phys = vox_to_phys(moving_image, moving_vox)

    # Displacement = where to sample in moving - where we are in fixed
    displacement = (moving_phys - fixed_phys).reshape(X, Y, Z, 3)
    displacement = np.transpose(displacement, (2, 1, 0, 3))


    disp_sitk = sitk.GetImageFromArray(displacement.astype(np.float64), isVector=True)
    disp_sitk.CopyInformation(fixed_image)

    df_transform = sitk.DisplacementFieldTransform(3)
    df_transform.SetDisplacementField(disp_sitk)

    return df_transform

def compose_e_resample(SlicerTPath, dfieldpath, fixed_image, moving_image):
    rigid = sitk.ReadTransform(SlicerTPath)
    dfield = pt_dfield_to_sitk(dfieldpath, fixed_image, moving_image)

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(dfield)
    composite.AddTransform(rigid)

    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        composite,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID()
    )
    return resampled

# resampled=compose_e_resample(SlicerTPath, dfieldpath, fixed_image, moving_image)
# sitk.WriteImage(resampled, '/Users/jbonaventura/Downloads/R23-055/ExVivo_MR/Block06Reg/TestOut332.nii.gz')
#



#
# def slicer_transform_from_h5(path):
#     with h5py.File(path, 'r') as f:
#         params = f['TransformGroup/0/TransformParameters'][:]
#         print(params)
#         center = f['TransformGroup/0/TransformFixedParameters'][:]
#         print(center)
#
#     M = params[:9].reshape(3, 3)
#     t = params[9:]
#
#     # Bake out center of rotation
#     offset = t + center - M @ center
#     print(offset)
#
#     # LPS -> RAS
#     L = np.diag([-1., -1., 1.])
#     mat4 = np.eye(4)
#     mat4[:3, :3] = L @ M @ L
#     mat4[:3,  3] = L @ offset
#
#     # Slicer shows the inverse direction
#     return np.linalg.inv(mat4)
#
# print(slicer_transform_from_h5(pathname))
#
