import os
import h5py
import numpy as np
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

# Applying Deformation Field-
def ApplyDfield(dfpath, movingvol, order=0):
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
    # Resample — order=0 nearest-neighbor preserves original voxel values; order=1 linear interpolates
    registered = map_coordinates(movingvol, coords, order=order, mode='constant', cval=0)
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

def ApplySlicerTransform(moving, fixedimpath, SlicerTPath, moving_affine=None, interpolator=sitk.sitkNearestNeighbor):
    fixed_image = sitk.ReadImage(fixedimpath)
    # Copy transform to local tmp path to avoid HDF5 file locking issues on network filesystems
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(SlicerTPath, tmp_path)
    try:
        transform = sitk.ReadTransform(tmp_path)
    finally:
        os.unlink(tmp_path)
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
        interpolator,  # interpolation method — sitkNearestNeighbor preserves values; sitkLinear interpolates
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

def propagate_tiles_to_space(all_corners, slice_num, RabbitID, Block, RabbitFolder, target_space='InVivo'):
    """
    Map tile corners from registered H&E / BlockFace pixel space back through the
    registration chain to a target space (InVivo, ExVivo, or ExVivoBlock).

    all_corners : (N_tiles, 4, 2) array of [row, col] in H&E registered pixel space
    slice_num   : int, z index in BlockFace NIfTI (from get_bf_slice_index)
    Returns     : (N_tiles, 4, 3) raw voxel coordinates in target_space for array indexing
    """
    from RabbitPathFinder import find_all_the_paths

    PROGRESSION  = ["InVivo", "ExVivo", "ExVivoBlock", "BlockFace"]
    all_steps    = ["ExVivoBlock", "ExVivo", "InVivo"]
    steps_to_run = all_steps[:3 - PROGRESSION.index(target_space)]

    N_tiles, N_corners, _ = all_corners.shape
    flat = all_corners.reshape(-1, 2)  # (N*4, 2) in [row, col]

    # Tile [row, col] → BlockFace raw NIfTI voxel [col/4, row/4, slice_num]
    raw_vox = np.column_stack([
        flat[:, 1] / 4.0,
        flat[:, 0] / 4.0,
        np.full(len(flat), float(slice_num)),
    ])

    if not steps_to_run:
        return raw_vox.reshape(N_tiles, N_corners, 3)

    # Raw BlockFace voxel → canonical BlockFace voxel
    bf_paths     = find_all_the_paths(RabbitID, Block, RabbitFolder, "ExVivoBlock")
    bf_nib       = nib.load(str(bf_paths['Fixed_FilePath']))
    bf_canonical = nib.as_closest_canonical(bf_nib)
    raw_hom      = np.column_stack([raw_vox, np.ones(len(raw_vox))])
    current_vox  = (np.linalg.inv(bf_canonical.affine) @ (bf_nib.affine @ raw_hom.T)).T[:, :3]
    current_affine = bf_canonical.affine

    moving_nib = moving_canonical = None

    for moving_key in steps_to_run:
        paths = find_all_the_paths(RabbitID, Block, RabbitFolder, moving_key)

        df_path = next(p for p in Path(paths['RegFold']).glob("*.pt") if not p.name.startswith("._"))
        field   = torch.load(str(df_path), map_location='cpu')['data'].numpy()  # [3, X, Y, Z]

        # Batch interpolate field at current canonical voxel coords
        coords = np.clip(current_vox.T, 0, np.array(field.shape[1:])[:, np.newaxis] - 1)
        displaced_vox = np.column_stack([
            map_coordinates(field[2], coords, order=1, mode='nearest') - 1,  # x
            map_coordinates(field[1], coords, order=1, mode='nearest') - 1,  # y
            map_coordinates(field[0], coords, order=1, mode='nearest') - 1,  # z
        ])  # (N*4, 3), still in current canonical space

        # Displaced voxel → physical RAS → physical LPS
        disp_hom = np.column_stack([displaced_vox, np.ones(len(displaced_vox))])
        phys_ras = (current_affine @ disp_hom.T).T[:, :3]
        phys_lps = phys_ras * np.array([-1., -1., 1.])

        # Apply Slicer transform (maps fixed→moving physical LPS)
        slicer_path = next(p for p in Path(paths['RegFold']).glob("*.h5") if not p.name.startswith("._"))
        T = sitk.ReadTransform(str(slicer_path))
        next_phys_lps = np.array([T.TransformPoint(p.tolist()) for p in phys_lps])

        # LPS → RAS → moving canonical voxel
        next_phys_ras    = next_phys_lps * np.array([-1., -1., 1.])
        moving_nib       = nib.load(str(paths['Moving_FilePath']))
        moving_canonical = nib.as_closest_canonical(moving_nib)
        next_hom         = np.column_stack([next_phys_ras, np.ones(len(next_phys_ras))])
        current_vox      = (np.linalg.inv(moving_canonical.affine) @ next_hom.T).T[:, :3]
        current_affine   = moving_canonical.affine

    # current_vox is in target_space canonical coords; convert to raw voxel for array indexing
    can_hom    = np.column_stack([current_vox, np.ones(len(current_vox))])
    phys_ras   = (moving_canonical.affine @ can_hom.T).T
    raw_target = (np.linalg.inv(moving_nib.affine) @ phys_ras.T).T[:, :3]

    return raw_target.reshape(N_tiles, N_corners, 3)


def propagate_tiles_to_Invivo_spaces(target_corners, RabbitID, RabbitFolder, Block):
    """
    Map tile corners from InVivo (End Day3 T1w) voxel space into all Start Day3 volumes
    (T1w structural, T1 map, T2 map, etc.).

    target_corners : (N_tiles, 4, 3) raw voxel coords in End Day3 T1w space
    Returns        : dict keyed by volume stem, each (N_tiles, 4, 3) voxel coords
    """
    import tempfile, shutil
    from RabbitPathFinder import find_all_the_paths, find_day3_paths

    invivo_paths = find_all_the_paths(RabbitID, Block, RabbitFolder, "InVivo")
    day3_paths   = find_day3_paths(RabbitID, RabbitFolder)

    N_tiles, N_corners, _ = target_corners.shape
    flat     = target_corners.reshape(-1, 3)
    flat_hom = np.column_stack([flat, np.ones(len(flat))])

    # End Day3 raw voxel → RAS → LPS
    end_nib  = nib.load(str(invivo_paths['Moving_FilePath']))
    phys_ras = (end_nib.affine @ flat_hom.T).T[:, :3]
    phys_lps = phys_ras * np.array([-1., -1., 1.])

    # Load cached inverse composite, or build and cache it
    cache_path = day3_paths['inv_transform_cache']
    fwd_path   = day3_paths['transform_path']

    def _load_transform(path):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(path, tmp_path)
        try:
            return sitk.ReadTransform(tmp_path)
        finally:
            os.unlink(tmp_path)

    if os.path.exists(cache_path):
        print('Loading cached inverse transform...')
        inv_composite = _load_transform(cache_path)
    else:
        print('Building inverse transform (first time — will be cached)...')
        fwd_composite = sitk.CompositeTransform(_load_transform(fwd_path))
        affine        = fwd_composite.GetNthTransform(0)
        disp_t        = sitk.DisplacementFieldTransform(fwd_composite.GetNthTransform(1))

        aff_params = np.array(affine.GetParameters())
        M_inv      = np.linalg.inv(aff_params[:9].reshape(3, 3))
        t_inv      = -M_inv @ aff_params[9:]
        inv_affine = sitk.AffineTransform(3)
        inv_affine.SetFixedParameters(affine.GetFixedParameters())
        inv_affine.SetParameters(list(M_inv.flatten()) + list(t_inv))

        print('Inverting displacement field...')
        inv_disp_t = sitk.DisplacementFieldTransform(
            sitk.InvertDisplacementField(disp_t.GetDisplacementField())
        )

        inv_composite = sitk.CompositeTransform(3)
        inv_composite.AddTransform(inv_disp_t)
        inv_composite.AddTransform(inv_affine)

        sitk.WriteTransform(inv_composite, cache_path)
        print(f'Cached to {cache_path}')

    # LPS End Day3 → LPS Start Day3
    start_phys_lps = np.array([inv_composite.TransformPoint(p.tolist()) for p in phys_lps])

    # LPS → RAS
    start_hom = np.column_stack([
        start_phys_lps * np.array([-1., -1., 1.]),
        np.ones(len(start_phys_lps))
    ])

    # Convert physical coords into each start/fixed vol's voxel space
    result = {}
    for vol_path in day3_paths['start_vols'] + day3_paths['fixed_vols']:
        vol_nib = nib.load(vol_path)
        vox     = (np.linalg.inv(vol_nib.affine) @ start_hom.T).T[:, :3]
        stem    = os.path.basename(vol_path).replace('.nii.gz', '')
        result[stem] = vox.reshape(N_tiles, N_corners, 3)

    return result


def propagate_tiles_to_day0(start_day3_corners, affine_field, splines_field, sdiff_field, n_iter=20):
    """
    Map tile corners from Start Day3 structural voxel space to Day0 voxel space by
    inverting the composed Amanpreet deformation fields via fixed-point iteration.

    Each field is an inverse warp: field[q] gives the source (Day3-side) coordinate
    for Day0 output voxel q.  Field shape: (3, X, Y, Z), channels field[0]=Z, field[1]=Y,
    field[2]=X, 0-indexed absolute voxel coordinates.

    The fields are applied in order affine → splines → sdiff (outermost to innermost),
    so the composed forward map is F(q) = affine[splines[sdiff[q]]].

    start_day3_corners : (N_tiles, 4, 3) voxel coords in Start Day3 structural T1w space
    affine_field       : Affine_deformation.npy as numpy array (3, X, Y, Z)
    splines_field      : SplinesProjection.npy as numpy array (3, X, Y, Z)
    sdiff_field        : sdiff.npy as numpy array (3, X, Y, Z)
    n_iter             : fixed-point iterations (20 is sufficient for typical deformations)
    Returns            : (N_tiles, 4, 3) voxel coords in Day0 structural space
    """
    from scipy.ndimage import map_coordinates

    def _sample(field, pts):
        coords = pts.T  # (3, N) = (X, Y, Z) indices into field spatial dims
        fx = map_coordinates(field[2], coords, order=1, mode='nearest')
        fy = map_coordinates(field[1], coords, order=1, mode='nearest')
        fz = map_coordinates(field[0], coords, order=1, mode='nearest')
        return np.stack([fx, fy, fz], axis=1)  # (N, 3) in XYZ

    p = start_day3_corners.reshape(-1, 3).astype(np.float64)
    q = p.copy()

    for _ in range(n_iter):
        src = _sample(affine_field, _sample(splines_field, _sample(sdiff_field, q)))
        q = q + (p - src)

    return q.reshape(start_day3_corners.shape)


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
