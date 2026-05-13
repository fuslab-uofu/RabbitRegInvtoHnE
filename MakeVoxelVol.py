import nibabel as nib
import numpy as np
import os

MOD = 12  # 6mm at 0.5mm voxel spacing — safe margin for continuous deformation


def make_voxel_label_nifti(reference_nifti_path, output_dir):
    ref = nib.load(reference_nifti_path)
    shape = ref.shape[:3]

    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    labels = (i % MOD) * MOD ** 2 + (j % MOD) * MOD + (k % MOD)

    print(labels[:10,:10])

    out = nib.Nifti1Image(labels.astype(np.float32), ref.affine)
    output_path = os.path.join(output_dir, "voxel_tile_vol.nii.gz")
    nib.save(out, output_path)



ref_nifti = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/48 3D_VIBE_0.5x0.5x1_cor_postContrast.nii'
out_path = '/Users/jbonaventura/Desktop/Annotations'
make_voxel_label_nifti(ref_nifti, out_path)