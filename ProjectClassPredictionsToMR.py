#Projects per-tile H&E class labels (HnE_Classification/prepare_class_labels.py) onto the
#MR grid.
#
#Each class channel is projected on its own (H&E tiles -> BlockFace via
#BuildBFromTile224Features, then BlockFace -> MR via the normalized adjoint init), so each
#MR voxel gets, per class, the coverage-weighted fraction of its H&E tiles labelled that
#class. All label handling (QC filtering, class calls) happens upstream in the label
#script; tiles it dropped are absent from the CSVs and are treated like off-tissue tiles.
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from BuildBFromTile224Features import build_b_and_Q_from_tile_features
from CombineBlockNiftis import block_count, block_paths, mean_over_blocks, stack_block_volumes
from NewtonAdjointInit import normalized_adjoint_init
from RabbitPathFinder import class_fraction_col, he_projection_dir, load_ready_rabbit_blocks

ANNOTATIONS_DIR = Path('/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/'
                       'HnE_Classification/Annotations')
# Which version of the per-tile labels to project (prepare_class_labels.py's LABEL_SOURCES):
# 'hard_<rule>' = one 0/1 class call per tile under that rule (prepare_class_labels.hard_label_rule,
# e.g. 'hard_immune0.6', 'hard_argmax'); 'prob_normalized' = probabilities divided by their sum;
# 'R3_softmax' = the R3 softmax model's probabilities as soft class fractions.
LABEL_SOURCE = 'R3_softmax'


def class_labels_dir(label_source):
    """Folder prepare_class_labels.py wrote label_source's per-tile CSVs to."""
    if label_source == 'prob_normalized':
        return ANNOTATIONS_DIR / 'tiff_224_class_probs_normalized'
    if label_source == 'R3_softmax':
        return ANNOTATIONS_DIR / 'tiff_224_class_probs_R3_softmax'
    if label_source.startswith('hard_'):
        return ANNOTATIONS_DIR / f"tiff_224_class_labels_{label_source.removeprefix('hard_')}"
    raise ValueError(f"Unknown LABEL_SOURCE {label_source!r}: use 'hard_<rule>', 'prob_normalized' "
                     f"or 'R3_softmax'")


CLASS_LABELS_DIR = class_labels_dir(LABEL_SOURCE)
BOX_COLS = ('x0', 'y0', 'x1', 'y1')


def class_channels(labels_dir):
    """Class channel names: every column of the label CSVs except the tile box."""
    first_csv = next(labels_dir.glob('*_pred.csv'))
    return [c for c in pd.read_csv(first_csv, nrows=0).columns if c not in BOX_COLS]


def project_rabbit_classes(RabbitID, Block, RabbitFolder, channels):
    """Project one rabbit block's class channels to the MR grid, one fraction map per class."""
    out_dir = str(Path(__file__).parent / 'TestData' /
                  f'{RabbitID}_Block{Block:02d}_ClassPredictions_{LABEL_SOURCE}')
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f'{RabbitID}_Block{Block:02d}')

    print(f"\n=== {RabbitID} Block{Block:02d}: {', '.join(channels)} ===")
    # One call for all classes: the registration and per-slide warps are loaded once.
    results, corners, source_shape, source_affine = build_b_and_Q_from_tile_features(
        RabbitID, Block, RabbitFolder, channels, feature_csv_dir=str(CLASS_LABELS_DIR))

    for i, class_name in enumerate(channels):
        b, Q, target_affine = results[(class_name, None)]

        b_path = f'{prefix}_b_{class_name}_fraction.nii.gz'
        nib.save(nib.Nifti1Image(b.astype(np.float32), target_affine), b_path)
        print(f"Saved BlockFace-grid fraction -> {b_path}")

        # Voxels with no H&E tiles under them are NaN (unknown), not 0, which would read
        # as "none of this class".
        a0, coverage = normalized_adjoint_init(b, Q, corners, source_shape,
                                               a0=np.full(source_shape, np.nan))
        mr_path = f'{prefix}_MR_{class_name}_fraction.nii.gz'
        nib.save(nib.Nifti1Image(a0.astype(np.float32), source_affine), mr_path)
        print(f"Saved MR-grid fraction -> {mr_path}")

        if i == 0:
            # Same tiles for every class, so the H&E coverage under each voxel is shared.
            coverage_path = f'{prefix}_MR_he_coverage.nii.gz'
            nib.save(nib.Nifti1Image(coverage.astype(np.float32), source_affine), coverage_path)
            print(f"Saved MR-grid H&E coverage -> {coverage_path}")


def combine_rabbit_blocks(rabbit_id, blocks, channels):
    """For viewing one multi-block rabbit's histology together (not used in feature extraction):
    a map of how many blocks have H&E in each voxel, and each class's fraction averaged over the
    blocks covering a voxel. Reads every block's saved maps, so they all must be projected."""
    maps = ['he_coverage'] + [class_fraction_col(c) for c in channels]
    missing = [p for m in maps for p in block_paths(rabbit_id, blocks, LABEL_SOURCE, m) if not os.path.exists(p)]
    if missing:
        print(f"Skipping R{rabbit_id} block combine, missing {len(missing)} map(s), e.g. {missing[0]}")
        return
    out_dir = he_projection_dir(LABEL_SOURCE)
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f'R{rabbit_id}_AllBlocks_MR')

    stack, affine = stack_block_volumes(block_paths(rabbit_id, blocks, LABEL_SOURCE, 'he_coverage'))
    count = block_count(stack)
    print(f"R{rabbit_id} blocks {blocks}, voxels by number of blocks with histology:",
          {int(n): int((count == n).sum()) for n in np.unique(count) if n > 0})
    nib.save(nib.Nifti1Image(count, affine), f'{prefix}_he_block_count.nii.gz')
    print(f"Saved block-overlap count -> {prefix}_he_block_count.nii.gz")

    for map_name in maps[1:]:
        stack, affine = stack_block_volumes(block_paths(rabbit_id, blocks, LABEL_SOURCE, map_name))
        out_path = f'{prefix}_{map_name}_block_mean.nii.gz'
        nib.save(nib.Nifti1Image(mean_over_blocks(stack), affine), out_path)
        print(f"Saved block-mean fraction -> {out_path}")


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    # Same rabbits/blocks as MSPredAndAnalysis.py ('R' prefix added below).
    RABBIT_BLOCKS = load_ready_rabbit_blocks()
    channels = class_channels(CLASS_LABELS_DIR)
    print(f"Class channels from {CLASS_LABELS_DIR}: {channels}")

    for rabbit_id, block in RABBIT_BLOCKS:
        project_rabbit_classes(f'R{rabbit_id}', block, RabbitFolder, channels)

    # Rabbits with several ready blocks also get their blocks combined (from the saved maps).
    blocks_by_rabbit = {}
    for rabbit_id, block in RABBIT_BLOCKS:
        blocks_by_rabbit.setdefault(rabbit_id, []).append(block)
    for rabbit_id, blocks in blocks_by_rabbit.items():
        if len(blocks) > 1:
            combine_rabbit_blocks(rabbit_id, blocks, channels)
