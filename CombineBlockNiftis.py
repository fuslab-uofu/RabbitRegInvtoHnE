"""Combine one rabbit's per-block MR-grid NIfTIs (e.g. the H&E class-fraction maps from
ProjectClassPredictionsToMR.py) into single volumes: a count of how many blocks have data in
each voxel, and the mean over blocks where they overlap."""
import os

import nibabel as nib
import numpy as np

from RabbitPathFinder import class_predictions_dir


def stack_block_volumes(paths):
    """(n_blocks, *grid) float stack of the volumes at paths, plus their shared affine. All must
    be on the same grid, since blocks of one rabbit are written onto that rabbit's MR grid."""
    imgs = [nib.load(str(p)) for p in paths]
    for p, img in zip(paths, imgs):
        if img.shape != imgs[0].shape or not np.allclose(img.affine, imgs[0].affine):
            raise ValueError(f"{p} is not on the same grid as {paths[0]}")
    return np.stack([img.get_fdata(dtype=np.float32) for img in imgs]), imgs[0].affine


def block_count(coverage_stack):
    """Number of blocks with histology in each voxel, from the he_coverage maps (0 where a block
    has no H&E, positive where it does): 0 = no histology, 1 = one block, 2+ = blocks overlap."""
    return (coverage_stack > 0).sum(axis=0).astype(np.uint8)


def mean_over_blocks(stack):
    """Mean over the blocks with data in each voxel; NaN where no block has data."""
    count = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def block_paths(rabbit_id, blocks, label_source, map_name):
    """Path of one MR-grid map (e.g. 'he_coverage', 'necrotic_tissue_fraction') for every block."""
    return [os.path.join(class_predictions_dir(rabbit_id, b, label_source),
                         f'R{rabbit_id}_Block{b:02d}_MR_{map_name}.nii.gz') for b in blocks]
