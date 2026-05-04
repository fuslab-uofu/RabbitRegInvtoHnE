import nibabel as nib
from ApplyTransforms import ApplySlicerTransform
import os
import csv
import glob
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import SimpleITK as sitk
from APS import AmanpreetsCode

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
orig_vol = '/Users/jbonaventura/Desktop/Annotations/Day0tiles/HnE_R23-055_H7_7a_5X_Day0_T1wCE_tile_overlay.nii.gz'
src_vol = '/Users/jbonaventura/Desktop/Annotations/Day3tiles/HnE_R23-055_H7_7a_5X_Day3_5_3d_vibe_05mmiso_cor_registration_tile_overlay.nii.gz'

save_path = os.path.join(os.path.dirname(day3_dir), 'Day3Transformed')
os.makedirs(save_path, exist_ok=True)

# vol_list = start_vols + fixed_vols + [resampled_end_path]
vol_list = glob.glob(os.path.join(day3_dir, '*.nii.gz'))

AmanpreetsCode.run_Day3_0_transforms(vol_list, transform_folder, orig_vol, src_vol, save_path)