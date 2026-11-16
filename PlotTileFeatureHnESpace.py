import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from BuildBFromTile224Features import (TILE_SIZE, DEFECT_MODEL_NAME, load_tile_feature_lookup,
                                       load_clean_prevalence, load_clean_q_lookup)

# Plots one 224px-tile QuPath feature in its own slide's H&E-native space (no BlockFace/TPS pull),
# before and after the same hard defect admission threshold BuildBFromTile224Features.py applies
# to b -- a direct look at what the defect map removes, without registration in the way.

Image.MAX_IMAGE_PIXELS = None  # normalized full-res TIFFs are legitimately huge (>1 billion px at level 0)


def load_hne_downsampled(tiff_path, downsample):
    """(rgb, full_w, full_h): the H&E TIFF at exactly 1/downsample of full-res, read from
    whichever pyramid level is closest (levels are fixed powers of two) and resized the
    rest of the way so it's pixel-aligned with paint_tile_feature's output."""
    with Image.open(tiff_path) as im:
        full_w, full_h = im.size
        level_widths = []
        for i in range(im.n_frames):
            im.seek(i)
            level_widths.append(im.size[0])
        best_level = int(np.argmin(np.abs(np.array(level_widths) / full_w - 1 / downsample)))
        im.seek(best_level)
        rgb = np.array(im.convert('RGB').resize((full_w // downsample, full_h // downsample)))
    return rgb, full_w, full_h


def paint_tile_feature(tile_values, tile_boxes, full_w, full_h, downsample, tile_size=TILE_SIZE):
    """Paints per-tile values (Series indexed by tile_id) onto a (full_h, full_w)/downsample
    grid: every output pixel takes the value of the tile its center falls in (no
    interpolation), NaN where there is no tile or the tile's value is NaN.

    The tile grid's origin is read from tile_boxes rather than assumed to be (0, 0) --
    Tile_Annotation_Tool's compute_tile_origins centers the grid on the slide, so every
    tile's x0/y0 carries the same nonzero remainder mod tile_size."""
    x_origin = tile_boxes['x0'].to_numpy() % tile_size
    y_origin = tile_boxes['y0'].to_numpy() % tile_size
    assert np.ptp(x_origin) == 0 and np.ptp(y_origin) == 0, "tile grid is not a single regular lattice"
    x_origin, y_origin = x_origin[0], y_origin[0]

    boxes = tile_boxes.set_index('tile_id')
    tile_cols = (boxes['x0'].to_numpy() - x_origin) // tile_size
    tile_rows = (boxes['y0'].to_numpy() - y_origin) // tile_size
    tile_grid = np.full((tile_rows.max() + 1, tile_cols.max() + 1), np.nan)
    tile_grid[tile_rows, tile_cols] = tile_values.reindex(boxes.index).to_numpy()

    x_full = (np.arange(full_w // downsample) + 0.5) * downsample
    y_full = (np.arange(full_h // downsample) + 0.5) * downsample
    px_cols = np.floor((x_full - x_origin) / tile_size).astype(np.int64)
    px_rows = np.floor((y_full - y_origin) / tile_size).astype(np.int64)
    cols_ok = (px_cols >= 0) & (px_cols < tile_grid.shape[1])
    rows_ok = (px_rows >= 0) & (px_rows < tile_grid.shape[0])

    painted = np.full((len(y_full), len(x_full)), np.nan)
    painted[np.ix_(rows_ok, cols_ok)] = tile_grid[np.ix_(px_rows[rows_ok], px_cols[cols_ok])]
    return painted


if __name__ == '__main__':
    ceph_base = '/System/Volumes/Data/ceph/hifu'
    RabbitFolder = os.path.join(ceph_base, 'users/jbonaventura/RabbitRegistrationProj/RabbitData')
    RabbitID = 'R24-103'
    Block = 6
    SLIDE_STEM = 'HnE_R24-103_H6_6a_5x_corrected_fullres'
    FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
    THRESHOLD = 0.7     # calibrated P(clean) admission cutoff -- tiles below it are excluded outright
    DOWNSAMPLE = 20     # full-res H&E pixels per plotted pixel
    CLIM_PERCENTILES = (1, 99)  # shared color limits, taken from the unfiltered map

    Block_str = f'Block{Block:02d}'
    analysis_dir = os.path.join(RabbitFolder, RabbitID, 'Analysis', Block_str)
    tiff_path = os.path.join(RabbitFolder, RabbitID, 'HnE', Block_str, 'NormalizedFull', f'{SLIDE_STEM}.tif')
    hne_classification_dir = os.path.join(ceph_base, 'users/jbonaventura/RabbitRegistrationProj/HnE_Classification')
    pred_csv_path = os.path.join(hne_classification_dir, 'Annotations', 'tiff_224_defect_predictions',
                                 f'{SLIDE_STEM}_pred.csv')
    defect_metrics_path = os.path.join(hne_classification_dir, 'trainedmodels', DEFECT_MODEL_NAME, 'metrics.json')

    tile_boxes = pd.read_csv(os.path.join(analysis_dir, f'tile_boxes_224px_{SLIDE_STEM}.csv'))
    tile_lookup = load_tile_feature_lookup(
        os.path.join(analysis_dir, f'{SLIDE_STEM}-cellcount-tile224px.csv'), FEATURE_COL)
    pi_clean = load_clean_prevalence(defect_metrics_path)
    q_lookup = load_clean_q_lookup(pred_csv_path, pi_clean)

    admitted = q_lookup.reindex(tile_lookup.index) >= THRESHOLD
    filtered_lookup = tile_lookup.where(admitted)
    print(f"{SLIDE_STEM}: {admitted.sum()}/{len(tile_lookup)} tiles admitted "
          f"at P(clean) >= {THRESHOLD} (pi_clean={pi_clean:.3f})")

    print(f"Loading H&E at 1/{DOWNSAMPLE} ...")
    hne_rgb, full_w, full_h = load_hne_downsampled(tiff_path, DOWNSAMPLE)
    unfiltered = paint_tile_feature(tile_lookup, tile_boxes, full_w, full_h, DOWNSAMPLE)
    filtered = paint_tile_feature(filtered_lookup, tile_boxes, full_w, full_h, DOWNSAMPLE)

    vmin, vmax = np.nanpercentile(unfiltered, CLIM_PERCENTILES)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True, constrained_layout=True)
    axes[0].imshow(hne_rgb)
    axes[0].set_title(SLIDE_STEM)
    axes[1].imshow(unfiltered, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title(f'{FEATURE_COL}\nno defect filtering')
    im = axes[2].imshow(filtered, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[2].set_title(f'{FEATURE_COL}\nP(clean) >= {THRESHOLD} ({admitted.mean():.1%} of tiles admitted)')
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02)
    plt.show()
