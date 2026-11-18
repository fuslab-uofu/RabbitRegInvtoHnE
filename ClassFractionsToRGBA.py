#Combines each block's H&E class-fraction maps on the MR grid (ProjectClassPredictionsToMR.py)
#into one RGBA NIfTI for viewing in Slicer over the MR: each class drives one colour channel,
#so mixed voxels show as blended colours, and voxels without H&E coverage get alpha 0.
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from RabbitPathFinder import class_fraction_col, class_predictions_dir, he_projection_dir, load_ready_rabbit_blocks


def fractions_to_rgba(fraction_paths):
    """RGBA32 NIfTI from three [0, 1] fraction maps, in R, G, B order (NaN = no H&E coverage).
    Each fraction scales its channel to 0-255; alpha is 255 where covered, 0 elsewhere."""
    images = [nib.load(str(p)) for p in fraction_paths]
    for im, path in zip(images[1:], fraction_paths[1:]):
        if im.shape != images[0].shape or not np.allclose(im.affine, images[0].affine):
            raise ValueError(f"{path} is not on the same grid as {fraction_paths[0]}")
    fractions = [im.get_fdata() for im in images]

    rgba = np.zeros(images[0].shape, dtype=[('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
    for channel, fraction in zip('RGB', fractions):
        rgba[channel] = np.round(np.clip(np.nan_to_num(fraction, nan=0.0), 0, 1) * 255)
    rgba['A'] = np.where(np.isfinite(fractions[0]), 255, 0)
    return nib.Nifti1Image(rgba, images[0].affine)


def save_rgba(fraction_paths, out_path):
    """fractions_to_rgba to out_path, skipping (with a message) if any fraction map is missing."""
    missing = [p for p in fraction_paths if not os.path.exists(p)]
    if missing:
        print(f"Skipping {Path(out_path).name}, missing {missing[0]}")
        return
    image = fractions_to_rgba(fraction_paths)
    nib.save(image, out_path)
    print(f"Saved {out_path} ({int((np.asarray(image.dataobj)['A'] > 0).sum()):,} covered voxels)")


if __name__ == '__main__':
    LABEL_SOURCE = 'R3_softmax'  # ProjectClassPredictionsToMR.py's LABEL_SOURCE
    # Class driving each colour channel, in R, G, B order.
    CHANNEL_CLASSES = ['necrotic_tissue', 'immune_cells', 'muscle']
    OUT_DIR = Path(__file__).parent / 'TestData' / f'ClassRGBA_{LABEL_SOURCE}'
    os.makedirs(OUT_DIR, exist_ok=True)

    blocks_by_rabbit = {}
    for rabbit_id, block_num in load_ready_rabbit_blocks():
        blocks_by_rabbit.setdefault(rabbit_id, []).append(block_num)
        prefix = f'R{rabbit_id}_Block{block_num:02d}_MR'
        block_dir = class_predictions_dir(rabbit_id, block_num, LABEL_SOURCE)
        save_rgba([block_dir / f'{prefix}_{class_fraction_col(c)}.nii.gz' for c in CHANNEL_CLASSES],
                  OUT_DIR / f'{prefix}_class_RGBA.nii.gz')

    # Multi-block rabbits: the block-mean fractions ProjectClassPredictionsToMR.combine_rabbit_blocks wrote.
    for rabbit_id, blocks in blocks_by_rabbit.items():
        if len(blocks) < 2:
            continue
        prefix = f'R{rabbit_id}_AllBlocks_MR'
        combined_dir = Path(he_projection_dir(LABEL_SOURCE))
        save_rgba([combined_dir / f'{prefix}_{class_fraction_col(c)}_block_mean.nii.gz' for c in CHANNEL_CLASSES],
                  OUT_DIR / f'{prefix}_class_RGBA.nii.gz')
