import nibabel as nib
from ApplyTransforms import ApplySlicerTransform
import os
import csv
import glob
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import SimpleITK as sitk
from APS import AmanpreetsCode
import numpy as np
from scipy.ndimage import map_coordinates
from RabbitPathFinder import find_day3_paths
from ApplyTransforms import propagate_tiles_to_day0

day3_dir = '/Users/jbonaventura/Desktop/Annotations/Day3tiles'
# log_path = os.path.join(day3_dir, 'R25_055_InVDay3Log.csv')
#
# # Read CSV and build stem -> category mapping
# stem_to_category = {}
# with open(log_path) as f:
#     for row in csv.DictReader(f):
#         stem_to_category[row['File Name'].strip()] = row['Home'].strip()
#
# # Sort niftis into start/end/fixed lists by matching stems
# start_vols, end_vols, fixed_vols = [], [], []
# for path in sorted(glob.glob(os.path.join(day3_dir, '*.nii.gz'))):
#     stem = os.path.basename(path).replace('.nii.gz', '')
#     category = stem_to_category.get(stem)
#     if category == 'Start':
#         start_vols.append(path)
#     elif category == 'End':
#         end_vols.append(path)
#     elif category == 'Fixed':
#         fixed_vols.append(path)
#
# day3_transform = os.path.join(day3_dir, 'Day3_end_to_start_Transform.h5')
#
# print('Fixed:')
# for v in fixed_vols:
#     print(' ', os.path.basename(v))
# print('Start of Day 3:')
# for v in start_vols:
#     print(' ', os.path.basename(v))
# print('End of Day 3:')
# for v in end_vols:
#     print(' ', os.path.basename(v))
# print('Transform:', os.path.basename(day3_transform))
#
#
# end_stem = os.path.basename(end_vols[0]).replace('.nii.gz', '')
# resampled_end_path = os.path.join(day3_dir, f'{end_stem}_reg_to_start.nii.gz')
#
# if not os.path.exists(resampled_end_path):
#     print('Applying Slicer transform for End Day 3...')
#     resampled_SlicerT = ApplySlicerTransform(end_vols[0], fixed_vols[0], day3_transform, interpolator=sitk.sitkLinear)
#     fixed_nib = nib.load(fixed_vols[0])
#     nib.save(nib.Nifti1Image(resampled_SlicerT, fixed_nib.affine, fixed_nib.header), resampled_end_path)
#     print(f'Saved to {resampled_end_path}')
# else:
#     print('Resampled End Day 3 already exists, skipping Slicer transform.')
#
# # Test inverse Slicer transform: resample a Start Day3 vol into End Day3 space
# print('Applying inverse Slicer transform for Start Day 3...')
# import tempfile, shutil
# import numpy as np
# with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
#     tmp_path = tmp.name
# shutil.copy2(day3_transform, tmp_path)
# try:
#     fwd_transform = sitk.ReadTransform(tmp_path)
# finally:
#     os.unlink(tmp_path)
#
# # Extract affine and displacement field components from the composite
# fwd_composite = sitk.CompositeTransform(fwd_transform)
# affine = fwd_composite.GetNthTransform(0)
# disp_transform = sitk.DisplacementFieldTransform(fwd_composite.GetNthTransform(1))
#
# # Invert affine analytically
# aff_params = np.array(affine.GetParameters())
# M = aff_params[:9].reshape(3, 3)
# t = aff_params[9:]
# M_inv = np.linalg.inv(M)
# t_inv = -M_inv @ t
# inv_affine = sitk.AffineTransform(3)
# inv_affine.SetFixedParameters(affine.GetFixedParameters())
# inv_affine.SetParameters(list(M_inv.flatten()) + list(t_inv))
#
# # Invert displacement field numerically (slow for large fields)
# print('Inverting displacement field (this may take a while)...')
# field_image = disp_transform.GetDisplacementField()
# inv_field_image = sitk.InvertDisplacementField(field_image)
# inv_disp_transform = sitk.DisplacementFieldTransform(inv_field_image)
#
# # Build inverse composite: reversed order (inv disp first, then inv affine)
# inv_composite = sitk.CompositeTransform(3)
# inv_composite.AddTransform(inv_disp_transform)
# inv_composite.AddTransform(inv_affine)
#
# moving_start = sitk.ReadImage(fixed_vols[0])
# fixed_end = sitk.ReadImage(end_vols[0])
# resampled_inv = sitk.Resample(moving_start, fixed_end, inv_composite, sitk.sitkLinear, 0.0, moving_start.GetPixelID())
# desktop_out = '/Users/jbonaventura/Desktop/start_reg_to_end_test.nii.gz'
# sitk.WriteImage(resampled_inv, desktop_out)
# print(f'Saved to {desktop_out}')

transform_folder = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/InVMRDataSets/Transforms"
orig_vol = '/Users/jbonaventura/Desktop/Annotations/Day0tiles/HnE_R23-055_H7_7a_5X_Day0_T1wCE_tile_overlay2.nii.gz'
src_vol = '/Users/jbonaventura/Desktop/Annotations/Day3tiles/HnE_R23-055_H7_7a_5X_Day3_5_3d_vibe_05mmiso_cor_registration_tile_overlay2.nii.gz'

save_path = os.path.join(os.path.dirname(day3_dir), 'Day3Transformed')
os.makedirs(save_path, exist_ok=True)

# vol_list = start_vols + fixed_vols + [resampled_end_path]
vol_list = glob.glob(os.path.join(day3_dir, '*.nii.gz'))

AmanpreetsCode.run_Day3_0_transforms(vol_list, transform_folder, orig_vol, src_vol, save_path)

# RABBIT_FOLDER = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
# RABBIT_ID     = 'R23-055'
#
#
# def _sample(field, pts):
#     coords = pts.T  # (3, N) — X, Y, Z indices into field spatial dims
#     # Fields store 1-indexed voxel coords (Amanpreet's convention); subtract 1 for 0-indexed.
#     fx = map_coordinates(field[2], coords, order=1, mode='nearest') - 1
#     fy = map_coordinates(field[1], coords, order=1, mode='nearest') - 1
#     fz = map_coordinates(field[0], coords, order=1, mode='nearest') - 1
#     return np.stack([fx, fy, fz], axis=1)  # (N, 3) XYZ
#
#
# def forward_F(pts, affine_field, splines_field, sdiff_field):
#     return _sample(affine_field, _sample(splines_field, _sample(sdiff_field, pts)))
#
#
# def diagnose_aps_field_inversion():
#     d3 = find_day3_paths(RABBIT_ID, RABBIT_FOLDER)
#     tf = d3['aps_transform_folder']
#
#     print(f"Loading fields from {tf}")
#     affine_field  = np.load(os.path.join(tf, 'Affine_deformation.npy'))
#     splines_field = np.load(os.path.join(tf, 'SplinesProjection.npy'))
#     sdiff_field   = np.load(os.path.join(tf, 'sdiff.npy'))
#
#     print(f"Field shapes: affine={affine_field.shape}, splines={splines_field.shape}, sdiff={sdiff_field.shape}")
#
#     # Sanity check: absolute voxel coords should be in volume bounds [0,384) x [0,144) x [0,512)
#     print("\nField value ranges (expect ~[0,384) X, ~[0,144) Y, ~[0,512) Z if absolute voxel coords):")
#     for name, field in [('affine', affine_field), ('splines', splines_field), ('sdiff', sdiff_field)]:
#         print(f"  {name}:  X(ch2) [{field[2].min():.1f}, {field[2].max():.1f}]"
#               f"  Y(ch1) [{field[1].min():.1f}, {field[1].max():.1f}]"
#               f"  Z(ch0) [{field[0].min():.1f}, {field[0].max():.1f}]")
#
#     # Round-trip: pick points in Day0 space, apply F forward -> Day3_start, then F^{-1} -> should recover original
#     # Volume shape (384, 144, 512) = (X, Y, Z)
#     test_pts = np.array([
#         [192,  72, 256],   # center
#         [ 96,  36, 128],   # quarter
#         [288, 108, 384],   # three-quarter
#         [ 50,  50,  50],   # near corner
#     ], dtype=np.float64)
#
#     p_day3 = forward_F(test_pts, affine_field, splines_field, sdiff_field)
#
#     print("\nForward map F (Day0 -> Day3_start):")
#     for q, p in zip(test_pts, p_day3):
#         print(f"  {q} -> {np.round(p, 2)}  (shift: {np.round(p - q, 2)})")
#
#     q_recovered = propagate_tiles_to_day0(
#         p_day3.reshape(1, -1, 3), affine_field, splines_field, sdiff_field
#     ).reshape(-1, 3)
#
#     print("\nRound-trip errors F^{-1}(F(q)) - q  [voxels]:")
#     for q0, qr in zip(test_pts, q_recovered):
#         err = np.linalg.norm(qr - q0)
#         print(f"  q0={q0}  recovered={np.round(qr, 2)}  |error|={err:.4f}")
#
#
# def diagnose_coordinate_chain():
#     from RabbitPathFinder import find_all_the_paths, find_day3_paths, find_day0_paths
#
#     invivo_paths = find_all_the_paths(RABBIT_ID, 7, RABBIT_FOLDER, "InVivo")
#     d3 = find_day3_paths(RABBIT_ID, RABBIT_FOLDER)
#     d0 = find_day0_paths(RABBIT_ID, RABBIT_FOLDER)
#
#     invivo_main = nib.load(str(invivo_paths['Moving_FilePath']))
#     print(f"InVivo main volume:  {invivo_paths['Moving_FilePath']}")
#     print(f"  shape={invivo_main.shape}  origin={np.round(invivo_main.affine[:3,3], 2)}  spacing={np.round(np.abs(np.diag(invivo_main.affine[:3,:3])), 3)}")
#
#     print(f"\nDay3 end_vols:")
#     for p in d3['end_vols']:
#         img = nib.load(p)
#         print(f"  {os.path.basename(p)}")
#         print(f"    shape={img.shape}  origin={np.round(img.affine[:3,3], 2)}  spacing={np.round(np.abs(np.diag(img.affine[:3,:3])), 3)}")
#
#     print(f"\nDay3 start_vols + fixed_vols:")
#     for p in d3['start_vols'] + d3['fixed_vols']:
#         img = nib.load(p)
#         print(f"  {os.path.basename(p)}")
#         print(f"    shape={img.shape}  origin={np.round(img.affine[:3,3], 2)}  spacing={np.round(np.abs(np.diag(img.affine[:3,:3])), 3)}")
#
#     print(f"\nDay0 vols:")
#     for p in d0['vols']:
#         img = nib.load(p)
#         print(f"  {os.path.basename(p)}")
#         print(f"    shape={img.shape}  origin={np.round(img.affine[:3,3], 2)}  spacing={np.round(np.abs(np.diag(img.affine[:3,:3])), 3)}")
#
#
# def diagnose_affines():
#     from RabbitPathFinder import find_all_the_paths, find_day3_paths
#
#     invivo_paths = find_all_the_paths(RABBIT_ID, 7, RABBIT_FOLDER, "InVivo")
#     d3 = find_day3_paths(RABBIT_ID, RABBIT_FOLDER)
#
#     invivo_main = nib.load(str(invivo_paths['Moving_FilePath']))
#     print(str(invivo_paths['Moving_FilePath']))
#     day3_end    = nib.load(d3['end_vols'][0])
#     print(d3['end_vols'][0])
#     day3_struct = nib.load(d3['fixed_vols'][0])  # structural used in propagate_tiles_to_day0
#     print(d3['fixed_vols'][0])
#
#     for label, img in [('InVivo main', invivo_main), ('Day3 end', day3_end), ('Day3 struct (fixed_vols[0])', day3_struct)]:
#         shape = img.shape
#         affine = img.affine
#         center_vox = np.array([s / 2 for s in shape] + [1.0])
#         center_ras = affine @ center_vox
#         print(f"\n{label}:")
#         print(f"  shape  : {shape}")
#         print(f"  affine :\n{np.round(affine, 3)}")
#         print(f"  center RAS: {np.round(center_ras[:3], 2)}")
#
#
# def diagnose_invivo_to_day3_chain():
#     """
#     Take the InVivo main center voxel, pass it through propagate_tiles_to_Invivo_spaces,
#     and check if the result lands near the expected Day3_struct position.
#
#     InVivo main center (192, 256, 72) in raw (i, j, k) space maps to physical RAS [-3, -56.25, -74].
#     Day3_struct voxel for that same physical location (without Slicer deformation) should be ~(192, 71, 255).
#     After Slicer inverse, it should be close to that.
#     """
#     from ApplyTransforms import propagate_tiles_to_Invivo_spaces
#     from RabbitPathFinder import find_day3_paths
#
#     # InVivo main center voxel in raw (i, j, k) coords — shape (384, 512, 144)
#     # i=192: X = -0.5*192 + 93 = -3
#     # j=256: Z = -0.5*256 + 54 = -74
#     # k=72:  Y = -0.5*72 + (-20.25) = -56.25
#     test_voxel = np.array([[[192., 256., 72.]]])  # (1 tile, 1 corner, 3)
#
#     day3_corners = propagate_tiles_to_Invivo_spaces(test_voxel, RABBIT_ID, RABBIT_FOLDER, 7)
#
#     d3 = find_day3_paths(RABBIT_ID, RABBIT_FOLDER)
#     structural_stem = os.path.basename(d3['fixed_vols'][0]).replace('.nii.gz', '')
#     print(f"\nStructural stem used for propagate_tiles_to_day0: {structural_stem}")
#     print(f"Day3_struct voxel for InVivo center {test_voxel[0,0]}:")
#     day3_vox = day3_corners[structural_stem][0, 0]
#     print(f"  result:                            {np.round(day3_vox, 2)}")
#     print(f"  expected (no Slicer deform): ~[192, 71, 255]")
#     print(f"  difference from expected:    {np.round(day3_vox - np.array([192., 71., 255.]), 2)}")
#
#     print(f"\nAll Day3 volume stems and resulting voxel coords:")
#     for stem, corners in day3_corners.items():
#         vox = corners[0, 0]
#         print(f"  {stem}: {np.round(vox, 2)}")
#
# diagnose_aps_field_inversion()
# diagnose_affines()
# diagnose_invivo_to_day3_chain()

