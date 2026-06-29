import os
import nibabel as nib
import numpy as np
from RabbitPathFinder import find_all_the_paths
from FinalRegProcessingPipeline import MultiStepReg

# ---------------------------------------------------------------------------
# Config — edit these per run
# ---------------------------------------------------------------------------
RABBIT_FOLDER = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RABBIT_ID     = 'R24-240'
BLOCK         = 3
END_FIXED     = 'BlockFace'

MOD = 12  # 6mm at 0.5mm voxel spacing — safe margin for continuous deformation


def make_and_register_voxel_vol(RabbitID, Block, RabbitFolder, end_fixed='BlockFace'):
    paths    = find_all_the_paths(RabbitID, Block, RabbitFolder, 'InVivo')
    ref_path = str(paths['Moving_FilePath'])
    proc_dir = os.path.join(paths['RegDataProc'], 'Voxel_Maps')
    os.makedirs(proc_dir, exist_ok=True)

    # Build voxel label volume in InVivo space
    ref   = nib.load(ref_path)
    shape = ref.shape[:3]
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    labels = (i % MOD) * MOD ** 2 + (j % MOD) * MOD + (k % MOD) + 1

    voxel_vol_path = os.path.join(proc_dir, 'voxel_tile_vol.nii.gz')
    nib.save(nib.Nifti1Image(labels.astype(np.float32), ref.affine), voxel_vol_path)
    print(f"Voxel label volume saved → {voxel_vol_path}")

    # Register InVivo → end_fixed (nearest-neighbour to preserve label values)
    print(f"Registering InVivo → {end_fixed}...")
    result, affine = MultiStepReg(
        RabbitID, Block, RabbitFolder,
        'InVivo', end_fixed,
        interpolation='nearest',
        moving_path=voxel_vol_path,
        save=False,
    )

    # Save registered result to RegDataProc/Voxel_Maps
    end_label = (end_fixed
                 .replace('ExVivoBlock', f'ExVivoBlock{Block:02d}')
                 .replace('BlockFace',   f'Block{Block:02d}'))
    out_path = os.path.join(proc_dir, f"voxel_tile_vol_RegTo{end_label}.nii.gz")
    nib.save(nib.Nifti1Image(result, affine), out_path)
    print(f"Registered voxel volume saved → {out_path}")

    return out_path


if __name__ == '__main__':
    make_and_register_voxel_vol(RABBIT_ID, BLOCK, RABBIT_FOLDER, END_FIXED)
