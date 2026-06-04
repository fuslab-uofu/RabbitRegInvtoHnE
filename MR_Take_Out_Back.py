#Import Libraries
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label, binary_fill_holes


#Import template image-
temp_path = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/InVMRDataSets/Day0/T1wCE.nii.gz"
tempvol = nib.load(temp_path).get_fdata()
print(tempvol.shape)

# Normalize to 0-255, boost above-threshold pixels, blur slice-by-slice to bridge cavities
thresh = 10
maskthresh = 150
kernsize = 15

pic = np.array(tempvol / np.max(tempvol) * 255).astype(np.uint8)
thresh_pic = np.where(pic > thresh, 1000, pic).astype(np.float32)

kernel = np.ones((kernsize, kernsize), np.float32) / (kernsize * kernsize)
filtered = np.zeros(thresh_pic.shape, dtype=np.float32)
for k in range(thresh_pic.shape[2]):
    filtered[:, :, k] = cv2.filter2D(thresh_pic[:, :, k], ddepth=-1, kernel=kernel)

mask = filtered > maskthresh

fig, axs = plt.subplots(1,3)
axs[0].imshow(pic[:,69,:])
axs[1].imshow(mask[:,69,:])
axs[2].imshow(pic[:,69,:]*mask[:,69,:])
plt.tight_layout()
plt.show()


# Apply mask and save for Slicer inspection
img = nib.load(temp_path)
masked_vol = img.get_fdata() * mask
out = nib.Nifti1Image(masked_vol.astype(np.float32), img.affine, img.header)
nib.save(out, '/Users/jbonaventura/Desktop/testout.nii.gz')
print('Saved testout.nii.gz')

# Apply mask to all volumes in the Day0 folder
import os
from pathlib import Path

data_dir = Path('/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/InVMRDataSets/Day0')
out_dir = Path('/Users/jbonaventura/Desktop/TestOutputs')

for nii_file in sorted(data_dir.glob('*.nii.gz')):
    img = nib.load(str(nii_file))
    masked_vol = img.get_fdata() * mask
    out = nib.Nifti1Image(masked_vol.astype(np.float32), img.affine, img.header)
    out_path = out_dir / nii_file.name
    nib.save(out, str(out_path))
    print(f'Saved {nii_file.name}')