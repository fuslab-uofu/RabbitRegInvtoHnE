import os
import math
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numbers_parser import Document

from RabbitPathFinder import (block_analysis_dir, block_label, class_fraction_col, class_predictions_dir,
                              day3end_mr_dir, load_ready_rabbit_blocks, mr_norm_csv_suffix)
from PaintHnEFeatureToInVivo import valid_mask, build_axial_summary
from PullMRFeaturesByVoxel import load_mr_volumes, extract_voxelwise_features, mr_grid_reference
from extract_mri_embedding import embed_tile, load_model, patch_pixel_center
from extract_mr_derived_features import gradient_magnitude, neighborhood_mean_std
from extract_radiomics_features import build_radiomics_extractor, radiomics_features

# Deliberately inclusive H&E slice threshold, used to compute tiling metadata
# (centroid/extent/z-range) from he_arr at load time.
HE_FRACTION_THRESHOLD = 0.0005


def reorient_to_grid(img, ref):
    """img's data on ref's voxel grid. The class maps are saved in canonical RAS while the MR
    volumes keep the scanner orientation, so this is an axis permutation/flip only; stops if the
    two still differ afterwards, i.e. they aren't the same grid."""
    transform = nib.orientations.ornt_transform(nib.io_orientation(img.affine),
                                                nib.io_orientation(ref.affine))
    out = img.as_reoriented(transform)
    if out.shape != ref.shape or not np.allclose(out.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{img.get_filename()} isn't on the MR grid {ref.shape} once reoriented")
    return out.get_fdata(dtype=np.float32)


def load_class_fraction_volumes(rabbit_id, block_num, label_source, mr_ref):
    """{class name: per-voxel fraction volume} from ProjectClassPredictionsToMR.py's MR-grid maps
    for label_source, on mr_ref's grid. NaN = no H&E covers the voxel."""
    pred_dir = class_predictions_dir(rabbit_id, block_num, label_source)
    prefix, suffix = f'R{rabbit_id}_Block{block_num:02d}_MR_', '_fraction.nii.gz'
    paths = sorted(pred_dir.glob(f'{prefix}*{suffix}'))
    if not paths:
        raise FileNotFoundError(f"No {prefix}*{suffix} maps in {pred_dir}")
    return {p.name[len(prefix):-len(suffix)]: reorient_to_grid(nib.load(p), mr_ref) for p in paths}


def classify_mr_column_day(column_name):
    """'day3' if the MR_renames header says 'Day 3', else 'day0' (including untagged
    columns like Adc Post/Pre and Max Temp Proj, which are Day 0)."""
    return 'day3' if 'Day 3' in column_name else 'day0'


def _read_rename_rows(rename_doc_path):
    """(header row, {rabbit_id: row}) from MR_renames.numbers."""
    doc = Document(rename_doc_path)
    header, rows = None, {}
    for sheet in doc.sheets:
        for table in sheet.tables:
            for row in table.rows(values_only=True):
                if header is None:
                    header = row
                    continue
                rows[row[0]] = row
    return header, rows


def columns_present_for_all_rabbits(rename_doc_path, rabbit_ids, day='day3'):
    """Contrast columns for `day` that are filled in for every rabbit, checked in the
    rename doc before any volume is loaded."""
    header, rows = _read_rename_rows(rename_doc_path)
    missing_rabbits = [rid for rid in rabbit_ids if rid not in rows]
    if missing_rabbits:
        raise ValueError(f"No row for {missing_rabbits} in {rename_doc_path}")

    day_cols = [col for col in header[1:]
                if day == 'all' or classify_mr_column_day(col) == day]
    present = {col for col in day_cols
               if all(rows[rid][header.index(col)] for rid in rabbit_ids)}
    dropped = set(day_cols) - present
    if dropped:
        print(f"Dropping {len(dropped)} contrast(s) not present for every rabbit "
              f"in the doc: {sorted(dropped)}")
    return present


def load_day_labels(rename_doc_path, rabbit_id, day='day3', allowed_columns=None):
    """{filename stem: standardized label} for one rabbit, limited to `day` ('day3'/'day0'/
    'all') and allowed_columns if given. Blank cells (contrast not acquired) are skipped."""
    header, rows = _read_rename_rows(rename_doc_path)
    if rabbit_id not in rows:
        raise ValueError(f"No row for '{rabbit_id}' in {rename_doc_path}")
    rabbit_row = rows[rabbit_id]

    wanted = {col for col in header[1:]
              if (day == 'all' or classify_mr_column_day(col) == day)
              and (allowed_columns is None or col in allowed_columns)}
    return {val: col for col, val in zip(header, rabbit_row) if col in wanted and val}


def select_day_volumes(mr_volumes, stem_to_label):
    """Keep volumes whose stem prefix-matches a rename-doc stem (either direction).
    Returns ({stem: volume}, {stem: label}), keyed by the original file stem."""
    selected, labels = {}, {}
    for stem, arr in mr_volumes.items():
        for doc_stem, label in stem_to_label.items():
            if stem == doc_stem or stem.startswith(doc_stem) or doc_stem.startswith(stem):
                selected[stem] = arr
                labels[stem] = label
                break
    missing = set(stem_to_label.values()) - set(labels.values())
    if missing:
        print(f"Note: label(s) with no matching loaded volume: {missing}")
    return selected, labels


def restrict_to_common_labels(per_block_labels):
    """{block name: {stem: label}} -> labels present for every block, so all blocks
    contribute the same contrasts."""
    label_sets = {name: set(labels.values()) for name, labels in per_block_labels.items()}
    common = set.intersection(*label_sets.values())
    varying = set.union(*label_sets.values()) - common
    print(f"{len(common)} MR contrast(s) common to all {len(per_block_labels)} block(s)")
    if varying:
        print(f"{len(varying)} contrast(s) vary across blocks, dropped: {sorted(varying)}")
        for name, names in label_sets.items():
            still_missing = common - names
            if still_missing:
                print(f"  {name} missing: {sorted(still_missing)}")
    return common


def filter_to_labels(selected, labels, keep_labels):
    return ({stem: arr for stem, arr in selected.items() if labels[stem] in keep_labels},
            {stem: lab for stem, lab in labels.items() if lab in keep_labels})


def rename_to_standardized_mr_columns(df, mr_labels):
    """Replace file stems in column names (bare or as a '{stem}_...' prefix) with standardized
    labels so rabbits share column names. Longest stem wins; each column is renamed once."""
    rename_map = {}
    for stem in sorted(mr_labels, key=len, reverse=True):
        label = mr_labels[stem]
        for col in df.columns:
            if col in rename_map:
                continue
            if col == stem or col.startswith(stem + '_'):
                rename_map[col] = label + col[len(stem):]
    return df.rename(columns=rename_map)


def extract_tile(arr, cx, cy, tile_size, fill=0):
    """tile_size x tile_size crop of a 2D plane centered at (cx, cy), padded with `fill` (the
    plane's no-data value: 0 for MR, NaN for H&E) where it runs past the plane's edge.
    Clamps explicitly: a bare slice would wrap on negative indices instead of padding."""
    half = tile_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + tile_size, y0 + tile_size

    tile = np.full((tile_size, tile_size), fill, dtype=arr.dtype)
    src_x0, src_x1 = max(x0, 0), min(x1, arr.shape[0])
    src_y0, src_y1 = max(y0, 0), min(y1, arr.shape[1])
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    tile[dst_x0:dst_x0 + (src_x1 - src_x0), dst_y0:dst_y0 + (src_y1 - src_y0)] = \
        arr[src_x0:src_x1, src_y0:src_y1]
    return tile


def round_half_up(x):
    """Round half up (builtin round() rounds half to even)."""
    return math.floor(x + 0.5)


def compute_grid_shape(extent, tile_size):
    """Tiles needed along one axis: 1 below 1.5*tile_size, 2 below 2.5*tile_size, etc."""
    return max(1, round_half_up(extent / tile_size))


def tile_centers(centroid, n, stride):
    """n evenly spaced tile centers along one axis, symmetric around centroid."""
    return [round_half_up(centroid + (i - (n - 1) / 2) * stride) for i in range(n)]


def sliding_tile_offsets(patch, patch_stride):
    """(dx, dy) tile shifts that move a tile's patch grid in steps of patch_stride along both
    in-plane axes; patch_stride == patch gives only the unshifted grid."""
    if patch % patch_stride:
        raise ValueError(f"patch_stride {patch_stride} must divide the patch size {patch}")
    shifts = range(0, patch, patch_stride)
    return [(dx, dy) for dx in shifts for dy in shifts]


def nonzero_mean(arr):
    """Mean over the nonzero pixels (0 = no data for MR); NaN if there are none."""
    valid = (arr != 0)
    return float(arr[valid].mean()) if valid.any() else np.nan


def build_tiles(he_arr, mr_volumes, metadata, tile_size, stride=None, center_offset=(0, 0),
                he_class_volumes=None):
    """One H&E + MR tile bundle per (slice, row, col), with the grid sized to the H&E extent.
    `stride` defaults to tile_size (non-overlapping); center_offset shifts the whole grid.
    he_class_volumes ({class: fraction volume}) adds each class's tile as 'he_class_tiles'."""
    if stride is None:
        stride = tile_size
    cx, cy = metadata['cx'] + center_offset[0], metadata['cy'] + center_offset[1]
    x_extent = metadata['xhi'] - metadata['xlo']
    y_extent = metadata['yhi'] - metadata['ylo']
    n_x = compute_grid_shape(x_extent, tile_size)
    n_y = compute_grid_shape(y_extent, tile_size)
    x_centers = tile_centers(cx, n_x, stride)
    y_centers = tile_centers(cy, n_y, stride)

    tiles = []
    for z in range(metadata['zlo'], metadata['zhi'] + 1):
        he_slice = he_arr[:, z, :]
        mr_slices = {name: vol[:, z, :] for name, vol in mr_volumes.items()}
        class_slices = {name: vol[:, z, :] for name, vol in (he_class_volumes or {}).items()}
        for row, tx in enumerate(x_centers):
            for col, ty in enumerate(y_centers):
                he_tile = extract_tile(he_slice, tx, ty, tile_size, fill=np.nan)
                mr_tiles = {name: extract_tile(sl, tx, ty, tile_size)
                            for name, sl in mr_slices.items()}
                class_tiles = {name: extract_tile(sl, tx, ty, tile_size, fill=np.nan)
                               for name, sl in class_slices.items()}
                tiles.append({'slice': z, 'row': row, 'col': col,
                              'tile_cx': tx, 'tile_cy': ty,
                              'he_tile': he_tile, 'mr_tiles': mr_tiles,
                              'he_class_tiles': class_tiles})
    return tiles


def filter_tiles_by_he_coverage(tiles, min_fraction):
    """Drop tiles whose H&E coverage (non-NaN fraction; 0 is a real value) fails min_fraction.
    At 0.0 this means "> 0" (at least one covered pixel), otherwise ">=" ."""
    kept = [t for t in tiles if _patch_coverage_ok(valid_mask(t['he_tile']).mean(), min_fraction)]
    print(f"{len(kept)}/{len(tiles)} tile(s) kept at H&E coverage threshold {min_fraction}")
    return kept


def filter_tiles_by_mr_coverage(tiles, min_fraction=0.0):
    """Drop tiles where any MR contrast's nonzero fraction (0 = no data for MR) fails
    min_fraction. At 0.0 this means "> 0" (at least one real pixel), otherwise ">=" ."""
    def contrast_ok(arr):
        frac = (arr != 0).mean()
        return frac > 0 if min_fraction <= 0 else frac >= min_fraction
    kept = [t for t in tiles if all(contrast_ok(arr) for arr in t['mr_tiles'].values())]
    print(f"{len(kept)}/{len(tiles)} tile(s) kept at MR coverage threshold "
          f"{min_fraction} (every contrast)")
    return kept


def run_feature_extraction(tiles, feature_extractor=None):
    """Apply feature_extractor(mr_tiles, he_tile) to each tile; with no extractor,
    return tiles unchanged."""
    if feature_extractor is None:
        return tiles
    return [{'slice': t['slice'], 'features': feature_extractor(t['mr_tiles'], t['he_tile'])}
            for t in tiles]


def scale_for_embedding(tile, divisor):
    """MR tile mapped into the [0, 1] range RadioDino requires: divided by one fixed per-contrast
    divisor (the same for every rabbit, so their intensities stay comparable), values above it
    clipped to 1."""
    return np.clip(tile / divisor, 0.0, 1.0)


def extract_mr_embeddings(tiles, divisors=None):
    """RadioDino embeddings, one batched call per contrast, flattened to one row per (tile, patch).
    divisors ({contrast stem: divisor}) scales a contrast into [0, 1] for embedding only; a
    contrast without one must already be in [0, 1].
    Returns (rows with tile_i/patch_row/patch_col/features, patch size in pixels)."""
    if not tiles:
        return [], None
    divisors = divisors or {}
    contrast_names = list(tiles[0]['mr_tiles'].keys())
    embedded = {}
    for name in contrast_names:
        start = time.time()
        arrays = [t['mr_tiles'][name] for t in tiles]
        if name in divisors:
            arrays = [scale_for_embedding(a, divisors[name]) for a in arrays]
        embedded[name] = embed_tile(arrays)
        elapsed = time.time() - start
        print(f"  Embedded {len(tiles):,} tile(s) for '{name}' in {elapsed:.1f}s "
              f"({elapsed / len(tiles):.3f}s/tile)")
    patch = next(iter(embedded.values()))['patch']
    grids_by_contrast = {name: out['grid'] for name, out in embedded.items()}
    gh, gw = next(iter(grids_by_contrast.values())).shape[1:3]

    rows = []
    for tile_i in range(len(tiles)):
        for pi in range(gh):
            for pj in range(gw):
                rows.append({
                    'tile_i': tile_i, 'patch_row': pi, 'patch_col': pj,
                    'features': {name: grids_by_contrast[name][tile_i, pi, pj]
                                 for name in contrast_names},
                })
    return rows, patch


def pick_representative_slices(tiles, n=5):
    """n unique slice indices spread evenly across the slices that have tiles."""
    zs = sorted({t['slice'] for t in tiles})
    idx = np.linspace(0, len(zs) - 1, min(n, len(zs))).round().astype(int)
    return [zs[i] for i in idx]


def plot_tile_bundles(tiles, labels=None, n=5, title=None):
    """Spot-check grid: one row per representative slice (its (0,0) tile, or any surviving one),
    one column for H&E plus each MR contrast. `labels` sets column titles only."""
    zs = pick_representative_slices(tiles, n)
    bundles = []
    for z in zs:
        z_tiles = [t for t in tiles if t['slice'] == z]
        bundles.append(next((t for t in z_tiles if t['row'] == 0 and t['col'] == 0), z_tiles[0]))
    contrast_names = list(bundles[0]['mr_tiles'].keys())
    panels = ['he_tile'] + contrast_names

    nrows, ncols = len(bundles), len(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for row, bundle in zip(axes, bundles):
        for ax, panel in zip(row, panels):
            if panel == 'he_tile':
                img, name = bundle['he_tile'], 'H&E'
            else:
                img, name = bundle['mr_tiles'][panel], (labels or {}).get(panel, panel)
            ax.imshow(img, cmap='gray', origin='lower')
            ax.set_title(f"{name}\nslice {bundle['slice']}", fontsize=8)
            ax.axis('off')

    fig.suptitle(title or 'Representative tiles across contrasts')
    fig.tight_layout()
    return fig


def plot_tile_grid_over_fov(he_arr, mr_volumes, tiles, z, tile_size, labels=None, title=None):
    """Full field of view at slice z with every tile's footprint outlined, one panel for H&E
    plus each MR contrast."""
    slice_tiles = [t for t in tiles if t['slice'] == z]
    if not slice_tiles:
        raise ValueError(f"No tile(s) at slice {z}")

    panels = [('H&E', he_arr[:, z, :])] + [
        ((labels or {}).get(name, name), vol[:, z, :]) for name, vol in mr_volumes.items()
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    half = tile_size // 2
    for ax, (name, img) in zip(axes[0], panels):
        ax.imshow(img, cmap='gray', origin='lower')
        for t in slice_tiles:
            x0, y0 = t['tile_cx'] - half, t['tile_cy'] - half
            ax.add_patch(mpatches.Rectangle((y0, x0), tile_size, tile_size,
                                             facecolor='none', edgecolor='red', linewidth=1))
        ax.set_title(f"{name}\nslice {z}, {len(slice_tiles)} tile(s)", fontsize=9)
        ax.axis('off')

    fig.suptitle(title or f'Tile grid over full FOV, slice {z}')
    fig.tight_layout()
    return fig


def plot_he_mr_slices(he_arr, mr_volumes, metadata, labels=None, n=3, margin=10, title=None,
                      patch_outlines=None):
    """Rows = n slices evenly spaced inside the H&E z-range (skipping the sparse end slices);
    columns = H&E then each MR contrast. Red outline on MR panels = H&E coverage.
    patch_outlines: {slice: [(x0, y0, size, color)]} squares drawn on every panel of that slice."""
    zs = np.linspace(metadata['zlo'], metadata['zhi'], n + 2)[1:-1].round().astype(int)
    x0, x1 = max(metadata['xlo'] - margin, 0), metadata['xhi'] + margin + 1
    y0, y1 = max(metadata['ylo'] - margin, 0), metadata['yhi'] + margin + 1
    # Sorted by label: file order follows scanner series numbers, which differ per rabbit.
    mr_panels = sorted((((labels or {}).get(name, name), vol) for name, vol in mr_volumes.items()),
                       key=lambda p: p[0])
    panels = [('H&E', he_arr)] + mr_panels

    fig, axes = plt.subplots(len(zs), len(panels), figsize=(3 * len(panels), 3 * len(zs)),
                             squeeze=False, layout='constrained')
    # One intensity range per column, shared by every plotted slice.
    limits = [(np.nanmin(vol[x0:x1, zs, y0:y1]), np.nanmax(vol[x0:x1, zs, y0:y1]))
              for _, vol in panels]
    images = [None] * len(panels)
    for row_axes, z in zip(axes, zs):
        he_valid = valid_mask(he_arr[x0:x1, z, y0:y1])
        for col, (ax, (name, vol)) in enumerate(zip(row_axes, panels)):
            is_he = vol is he_arr
            vmin, vmax = limits[col]
            images[col] = ax.imshow(vol[x0:x1, z, y0:y1], cmap='viridis' if is_he else 'gray',
                                    origin='lower', vmin=vmin, vmax=vmax)
            if not is_he and he_valid.any() and not he_valid.all():
                ax.contour(he_valid.astype(float), levels=[0.5], colors='red', linewidths=0.6)
            for px0, py0, size, color in (patch_outlines or {}).get(z, []):
                # -0.5: imshow centers pixel i at coordinate i, so its edge is at i - 0.5.
                ax.add_patch(mpatches.Rectangle((py0 - y0 - 0.5, px0 - x0 - 0.5), size, size,
                                                facecolor='none', edgecolor=color, linewidth=0.7))
            ax.set_title(f"{name}\nslice {z}", fontsize=8)
            ax.axis('off')

    for col, im in enumerate(images):
        cbar = fig.colorbar(im, ax=axes[:, col], location='bottom', fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title or 'H&E vs MR slices')
    return fig


def sliding_patch_outlines(data, tile_size, patch, offsets, min_he_fraction, min_mr_fraction,
                           min_patch_he_fraction=0.0, min_patch_mr_fraction=0.0):
    """Outlines of the patches Stage 2 would keep at each tile shift, one color per shift.
    Returns ({slice: [(x0, y0, patch, color)]}, legend handles), for plot_he_mr_slices."""
    colors = (plt.cm.tab10 if len(offsets) <= 10 else plt.cm.tab20)(np.arange(len(offsets)))
    outlines, handles = {}, []
    for offset, color in zip(offsets, colors):
        tiles = build_tiles(data['he_arr'], data['mr_volumes'], data['metadata'], tile_size,
                            center_offset=offset)
        tiles = filter_tiles_by_he_coverage(tiles, min_he_fraction)
        tiles = filter_tiles_by_mr_coverage(tiles, min_mr_fraction)
        handles.append(mpatches.Patch(facecolor='none', edgecolor=color, label=f'shift {offset}'))
        half = tile_size // 2
        for tile in tiles:
            for pi in range(tile_size // patch):
                for pj in range(tile_size // patch):
                    if patch_coverage(tile, patch, pi, pj, min_patch_he_fraction,
                                      min_patch_mr_fraction)['kept']:
                        outlines.setdefault(tile['slice'], []).append(
                            (tile['tile_cx'] - half + pi * patch,
                             tile['tile_cy'] - half + pj * patch, patch, color))
    return outlines, handles


def plot_gradient_comparison(rabbit_data, key, z=None):
    """Debug plot: each contrast (top) above its in-plane gradient magnitude (bottom) at
    slice z (default: middle of the H&E z-range) for the (rid, block_num) block `key`."""
    data = rabbit_data[key]
    if z is None:
        z = (data['metadata']['zlo'] + data['metadata']['zhi']) // 2
    contrast_names = list(data['mr_volumes'].keys())
    fig, axes = plt.subplots(2, len(contrast_names),
                              figsize=(3 * len(contrast_names), 6), squeeze=False)

    for col, name in enumerate(contrast_names):
        arr = data['mr_volumes'][name]
        grad = gradient_magnitude(arr[:, z:z + 1, :])[:, 0, :]
        label = data['mr_labels'].get(name, name)

        axes[0, col].imshow(arr[:, z, :], cmap='gray', origin='lower')
        axes[0, col].set_title(f"{label}\n(original)", fontsize=9)
        axes[0, col].axis('off')

        axes[1, col].imshow(grad, cmap='gray', origin='lower')
        axes[1, col].set_title(f"{label}\n(gradient)", fontsize=9)
        axes[1, col].axis('off')

    fig.suptitle(f"{block_label(*key)} -- original vs. gradient magnitude, slice {z}")
    fig.tight_layout()
    return fig


GRADIENT_AXIS_VARIANTS = [
    ('axial', (0, 2)),      # in-plane, matches this pipeline's per-slice default
    ('coronal', (0, 1)),
    ('sagittal', (1, 2)),
    ('3d', (0, 1, 2)),
]


def gradient_extra_volumes(data):
    """Gradient magnitude of every contrast for each GRADIENT_AXIS_VARIANTS entry.
    Columns: '{contrast}_grad_{axis_label}'."""
    extra = {}
    for name, arr in data['mr_volumes'].items():
        for label, axes in GRADIENT_AXIS_VARIANTS:
            extra[f'{name}_grad_{label}'] = gradient_magnitude(arr, axes=axes)
    return extra


# Neighborhood radii in voxels: [1,3,5,7] in-plane (2D), [1,3] in 3D (capped near Stage 2's
# patch size). Applied to raw MR contrasts only, not to gradient maps.
NEIGHBORHOOD_RADII = {
    (0, 2): [1, 3, 5, 7],
    (0, 1, 2): [1, 3],
}
NEIGHBORHOOD_AXIS_LABELS = {(0, 2): '2d', (0, 1, 2): '3d'}


def neighborhood_extra_volumes(data):
    """Neighborhood mean/std of every contrast at each NEIGHBORHOOD_RADII (axes, radius).
    Columns: '{contrast}_nbhd_{mean|std}_{2d|3d}_r{radius}'."""
    extra = {}
    for name, arr in data['mr_volumes'].items():
        for axes, radii in NEIGHBORHOOD_RADII.items():
            axis_label = NEIGHBORHOOD_AXIS_LABELS[axes]
            for radius in radii:
                mean, std = neighborhood_mean_std(arr, radius, axes=axes)
                extra[f'{name}_nbhd_mean_{axis_label}_r{radius}'] = mean
                extra[f'{name}_nbhd_std_{axis_label}_r{radius}'] = std
    return extra


def combined_extra_volumes(data):
    """All derived feature volumes (gradient + neighborhood stats) for voxelwise_extraction."""
    extra = {}
    extra.update(gradient_extra_volumes(data))
    extra.update(neighborhood_extra_volumes(data))
    return extra


def load_rabbit_mr(rabbit_id, block_num, base_dir, rename_doc_path, label_source, day='day3',
                    mr_norm_dir='Normalized', allowed_columns=None, show_slices=False):
    """One block's MR volumes and H&E class fraction maps (label_source's projection), all on
    the MR grid. Returns (he_arr, he_classes, metadata, MR volumes, MR labels); he_arr is the
    H&E coverage map (1 where H&E covers the voxel, NaN elsewhere), which drives tiling and
    coverage, and he_classes holds the per-class fractions used as targets."""
    mr_dir = day3end_mr_dir(base_dir, rabbit_id, mr_norm_dir)
    mr_volumes = load_mr_volumes(mr_dir)

    he_classes = load_class_fraction_volumes(rabbit_id, block_num, label_source,
                                             mr_grid_reference(mr_dir))
    # Every class map is NaN exactly where no H&E covers the voxel, so any one gives coverage.
    he_arr = np.where(np.isfinite(next(iter(he_classes.values()))), 1.0, np.nan).astype(np.float32)
    metadata, _, _ = build_axial_summary(he_arr, HE_FRACTION_THRESHOLD)

    stem_to_label = load_day_labels(rename_doc_path, f'R{rabbit_id}', day=day,
                                     allowed_columns=allowed_columns)
    selected, labels = select_day_volumes(mr_volumes, stem_to_label)
    name = block_label(rabbit_id, block_num)
    print(f"{name}: {len(selected)}/{len(stem_to_label)} requested "
          f"{day} contrast(s) loaded")
    if show_slices:
        plot_he_mr_slices(he_arr, selected, metadata, labels=labels,
                          title=f"{name}: H&E vs MR ({mr_norm_dir or 'raw'})")
    return he_arr, he_classes, metadata, selected, labels


def load_all_rabbit_data(rabbit_blocks, base_dir, rename_doc_path, label_source, day='day3',
                          mr_norm_dir='Normalized', show_slices=False, contrasts=None):
    """Load every (rid, block_num) block's H&E/MR once, restricted to contrasts common to all
    blocks (and to `contrasts`, a list of standardized labels, if given).
    Returns {(rid, block_num): data}; blocks of one rabbit share that rabbit's MR volumes."""
    allowed_columns = columns_present_for_all_rabbits(
        rename_doc_path, sorted({f'R{rid}' for rid, _ in rabbit_blocks}), day=day)
    if contrasts is not None:
        unavailable = set(contrasts) - allowed_columns
        if unavailable:
            raise ValueError(f"Requested contrast(s) not available for every rabbit: {sorted(unavailable)}")
        allowed_columns = set(contrasts)

    per_block = {(rid, block_num): load_rabbit_mr(rid, block_num, base_dir, rename_doc_path,
                                                  label_source, day, mr_norm_dir, allowed_columns,
                                                  show_slices)
                 for rid, block_num in rabbit_blocks}

    common_labels = restrict_to_common_labels(
        {block_label(*key): labels for key, (*_, labels) in per_block.items()})

    rabbit_data = {}
    for key, (he_arr, he_classes, metadata, selected, labels) in per_block.items():
        selected, labels = filter_to_labels(selected, labels, common_labels)
        rabbit_data[key] = {
            'he_arr': he_arr, 'he_classes': he_classes, 'metadata': metadata,
            'mr_volumes': selected, 'mr_labels': labels,
        }
    return rabbit_data


def build_all_rabbit_tiles(rabbit_data, tile_size=224, min_he_fraction=None, min_mr_fraction=None,
                           center_offset=(0, 0)):
    """Tile already-loaded block data at tile_size, applying the coverage thresholds if given."""
    all_tiles = {}
    for key, data in rabbit_data.items():
        tiles = build_tiles(data['he_arr'], data['mr_volumes'], data['metadata'], tile_size,
                            center_offset=center_offset, he_class_volumes=data['he_classes'])
        if min_he_fraction is not None:
            tiles = filter_tiles_by_he_coverage(tiles, min_he_fraction)
        if min_mr_fraction is not None:
            tiles = filter_tiles_by_mr_coverage(tiles, min_mr_fraction)
        all_tiles[key] = tiles
    return all_tiles


def voxelwise_extraction(rabbit_data, base_dir, extra_volumes_fn=None,
                          csv_name='voxel_features_by_id.csv'):
    """Stage 0/1a: one row per voxel with valid H&E and nonzero MR in every contrast (targets =
    each class's fraction at the voxel), plus extra_volumes_fn(data) derived columns (not used
    for masking) -> csv_name."""
    for (rid, block_num), data in rabbit_data.items():
        extra_volumes = extra_volumes_fn(data) if extra_volumes_fn is not None else None
        targets = {class_fraction_col(c): vol for c, vol in data['he_classes'].items()}
        merged = extract_voxelwise_features(targets, data['mr_volumes'], extra_volumes=extra_volumes)
        merged = rename_to_standardized_mr_columns(merged, data['mr_labels'])
        out_path = os.path.join(block_analysis_dir(base_dir, rid, block_num), csv_name)
        merged.to_csv(out_path, index=False)
        print(f"{block_label(rid, block_num)}: saved {len(merged):,} row(s) -> {out_path}")


def _patch_coverage_ok(frac, min_fraction):
    """Coverage test: '> 0' at a 0.0 threshold, otherwise '>= min_fraction'."""
    return frac > 0 if min_fraction <= 0 else frac >= min_fraction


def _patch_slice(tile, patch, pi, pj):
    """(he_sub, mr_subs, class_subs): patch (pi, pj)'s sub-crop of a tile."""
    r0, r1 = pi * patch, (pi + 1) * patch
    c0, c1 = pj * patch, (pj + 1) * patch
    he_sub = tile['he_tile'][r0:r1, c0:c1]
    mr_subs = {name: arr[r0:r1, c0:c1] for name, arr in tile['mr_tiles'].items()}
    class_subs = {name: arr[r0:r1, c0:c1] for name, arr in tile.get('he_class_tiles', {}).items()}
    return he_sub, mr_subs, class_subs


def patch_coverage(tile, patch, pi, pj, min_patch_he_fraction, min_patch_mr_fraction):
    """Patch (pi, pj)'s H&E/MR sub-crops, coverage fractions, 'kept' flag at the given
    thresholds, and 'class_means' = each class's plain mean fraction over the patch's
    H&E-covered voxels (NaN if none). Shared by the extraction stages and the preview plots."""
    he_sub, mr_subs, class_subs = _patch_slice(tile, patch, pi, pj)
    he_valid = valid_mask(he_sub)
    he_frac = float(he_valid.mean())
    mr_fracs = {name: float((arr != 0).mean()) for name, arr in mr_subs.items()}
    he_ok = _patch_coverage_ok(he_frac, min_patch_he_fraction)
    mr_ok = all(_patch_coverage_ok(f, min_patch_mr_fraction) for f in mr_fracs.values())
    class_means = {name: float(arr[he_valid].mean()) if he_valid.any() else np.nan
                   for name, arr in class_subs.items()}
    return {
        'he_sub': he_sub, 'he_valid': he_valid, 'mr_subs': mr_subs,
        'he_frac': he_frac, 'mr_fracs': mr_fracs, 'class_means': class_means,
        'kept': he_ok and mr_ok,
    }


def _representative_tiles(tiles, n):
    """One tile per representative slice: its (0,0) tile, or any surviving one."""
    zs = pick_representative_slices(tiles, n)
    bundles = []
    for z in zs:
        z_tiles = [t for t in tiles if t['slice'] == z]
        bundles.append(next((t for t in z_tiles if t['row'] == 0 and t['col'] == 0), z_tiles[0]))
    return bundles


def plot_patch_coverage_grid(tiles, patch, min_patch_he_fraction, min_patch_mr_fraction,
                              labels=None, n=5, title=None):
    """Representative tiles with patch grid drawn and green over patches that would pass the
    patch coverage filter, for tuning thresholds without running the embedding."""
    bundles = _representative_tiles(tiles, n)
    contrast_names = list(bundles[0]['mr_tiles'].keys())
    panels = ['he_tile'] + contrast_names
    nrows, ncols = len(bundles), len(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for row_axes, bundle in zip(axes, bundles):
        gh = bundle['he_tile'].shape[0] // patch
        gw = bundle['he_tile'].shape[1] // patch
        kept = np.zeros((gh, gw), dtype=bool)
        for pi in range(gh):
            for pj in range(gw):
                kept[pi, pj] = patch_coverage(bundle, patch, pi, pj,
                                               min_patch_he_fraction, min_patch_mr_fraction)['kept']
        overlay_mask = kept.repeat(patch, axis=0).repeat(patch, axis=1)

        for ax, panel in zip(row_axes, panels):
            if panel == 'he_tile':
                img, name = bundle['he_tile'], 'H&E'
            else:
                img, name = bundle['mr_tiles'][panel], (labels or {}).get(panel, panel)
            ax.imshow(img, cmap='gray', origin='lower')
            overlay = np.zeros((*img.shape, 4))
            overlay[overlay_mask] = (0, 1, 0, 0.3)
            ax.imshow(overlay, origin='lower')
            for pi in range(1, gh):
                ax.axhline(pi * patch, color='white', linewidth=0.3)
            for pj in range(1, gw):
                ax.axvline(pj * patch, color='white', linewidth=0.3)
            ax.set_title(f"{name}\nslice {bundle['slice']} ({kept.sum()}/{kept.size} kept)",
                         fontsize=8)
            ax.axis('off')

    fig.suptitle(title or 'Patch coverage preview (green = would survive patch-level filter)')
    fig.tight_layout()
    return fig


def plot_patch_samples(tiles, patch, min_patch_he_fraction, min_patch_mr_fraction,
                        labels=None, n=5, n_slices=5, title=None, rng_seed=42):
    """Random sample of n patches that pass the coverage filter, one row per patch,
    one column for H&E plus each MR contrast."""
    bundles = _representative_tiles(tiles, n_slices)

    candidates = []
    for tile in bundles:
        gh = tile['he_tile'].shape[0] // patch
        gw = tile['he_tile'].shape[1] // patch
        for pi in range(gh):
            for pj in range(gw):
                cov = patch_coverage(tile, patch, pi, pj,
                                      min_patch_he_fraction, min_patch_mr_fraction)
                if cov['kept']:
                    candidates.append((tile, cov))

    print(f"{len(candidates):,} patch(es) would survive coverage filtering across "
          f"{len(bundles):,} representative tile(s)")
    if not candidates:
        return None
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    chosen = [candidates[i] for i in idx]

    contrast_names = list(chosen[0][1]['mr_subs'].keys())
    panels = ['he'] + contrast_names
    nrows, ncols = len(chosen), len(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for row_axes, (tile, cov) in zip(axes, chosen):
        for ax, panel in zip(row_axes, panels):
            if panel == 'he':
                img, name, frac = cov['he_sub'], 'H&E', cov['he_frac']
            else:
                img = cov['mr_subs'][panel]
                name, frac = (labels or {}).get(panel, panel), cov['mr_fracs'][panel]
            ax.imshow(img, cmap='gray', origin='lower')
            ax.set_title(f"{name}\nslice {tile['slice']}, {frac:.0%} covered", fontsize=8)
            ax.axis('off')

    fig.suptitle(title or f'Sample of {len(chosen)} patch(es) that would survive coverage filtering')
    fig.tight_layout()
    return fig


def FMTilewiseExtraction(rabbit_data, base_dir, tile_size=224,
                          min_he_fraction=0.1, min_mr_fraction=0.1,
                          min_patch_he_fraction=0.0, min_patch_mr_fraction=0.0,
                          patch_stride=None, csv_suffix='', embedding_divisors=None):
    """Stage 2: RadioDino embeddings per 16x16 patch, filtered by coverage AFTER embedding
    (ViT attention mixes the whole tile). Targets = each class's mean fraction over the patch's
    H&E-covered voxels -> CSV.
    patch_stride slides the patch grid: the tiles are re-embedded at every in-plane shift of
    that many voxels and all shifts are pooled into their own CSV. None = one unshifted grid.
    csv_suffix is appended to the CSV name (see RabbitPathFinder.mr_norm_csv_suffix).
    embedding_divisors ({standardized contrast label: divisor}) scales those contrasts into
    [0, 1] for the embedding only; the patch-mean MR columns keep the unscaled values."""
    model_patch = load_model()[1]['patch']
    offsets = sliding_tile_offsets(model_patch, patch_stride or model_patch)
    csv_name = ('patch_embeddings_by_id' if patch_stride is None else
                f'patch_embeddings_by_id_stride{patch_stride}') + f'{csv_suffix}.csv'

    for (rid, block_num), data in rabbit_data.items():
        block_name = block_label(rid, block_num)
        stem_divisors = {stem: embedding_divisors[label] for stem, label in data['mr_labels'].items()
                         if label in (embedding_divisors or {})}
        rabbit_start = time.time()
        # One table per shift, so only one shift's embeddings are in memory at a time.
        frames = []
        n_patches = 0
        for offset in offsets:
            tiles = build_all_rabbit_tiles({block_name: data}, tile_size, min_he_fraction,
                                           min_mr_fraction, center_offset=offset)[block_name]
            if not tiles:
                continue

            n_contrasts = len(tiles[0]['mr_tiles'])
            print(f"{block_name}, shift {offset}: embedding {len(tiles):,} tile(s) across "
                  f"{n_contrasts} contrast(s)...")
            patch_rows, patch = extract_mr_embeddings(tiles, stem_divisors)
            n_patches += len(patch_rows)

            records = []
            for row in patch_rows:
                tile = tiles[row['tile_i']]
                pi, pj = row['patch_row'], row['patch_col']
                cov = patch_coverage(tile, patch, pi, pj, min_patch_he_fraction, min_patch_mr_fraction)
                if not cov['kept']:
                    continue

                px, py = patch_pixel_center(tile['tile_cx'], tile['tile_cy'], tile_size, patch, pi, pj)
                z = tile['slice']
                record = {
                    'vx': int(px), 'vy': int(py), 'vz': int(z),
                    **{class_fraction_col(c): m for c, m in cov['class_means'].items()},
                    # Saved so barely-covered patches can be down-weighted downstream.
                    'he_coverage_frac': cov['he_frac'],
                }
                for name, frac in cov['mr_fracs'].items():
                    record[f'{name}_coverage_frac'] = frac
                # Patch-mean raw MR: the intensity-only baseline on the same rows as the embedding.
                for name, arr in cov['mr_subs'].items():
                    record[name] = nonzero_mean(arr)
                for name, vec in row['features'].items():
                    for d, val in enumerate(vec):
                        record[f'{name}_emb{d}'] = float(val)
                records.append(record)
            if records:
                frames.append(pd.DataFrame(records))

        if not frames:
            print(f"{block_name}: no patch(es) survived coverage filtering, skipping")
            continue

        merged = pd.concat(frames, ignore_index=True)
        print(f"{block_name}: {len(merged):,}/{n_patches:,} patch(es) kept "
              f"(rest dropped for empty H&E/MR coverage)")
        merged = rename_to_standardized_mr_columns(merged, data['mr_labels'])
        out_path = os.path.join(block_analysis_dir(base_dir, rid, block_num), csv_name)
        merged.to_csv(out_path, index=False)
        print(f"{block_name}: saved {len(merged):,} row(s) -> {out_path} "
              f"(block total: {time.time() - rabbit_start:.1f}s)")


def tilewise_aggregate_extraction(rabbit_data, base_dir, tile_sizes,
                                   min_he_fraction, min_mr_fraction, csv_suffix=''):
    """Stage 1b: per tile, each class's mean fraction over its H&E-covered voxels (targets), mean
    of each raw MR contrast, and PyRadiomics features.
    One CSV per tile_size, since each size defines its own tile grid. csv_suffix is appended
    to each CSV name (see RabbitPathFinder.mr_norm_csv_suffix)."""
    radiomics_extractor = build_radiomics_extractor()

    for tile_size in tile_sizes:
        all_tiles = build_all_rabbit_tiles(rabbit_data, tile_size, min_he_fraction, min_mr_fraction)

        for (rid, block_num), tiles in all_tiles.items():
            block_name = block_label(rid, block_num)
            if not tiles:
                print(f"{block_name} (tile_size={tile_size}): no tile(s) survived "
                      f"tile-level coverage filtering, skipping")
                continue

            records = []
            for tile in tiles:
                cov = patch_coverage(tile, tile_size, 0, 0, min_he_fraction, min_mr_fraction)
                if not cov['kept']:
                    continue

                px, py = patch_pixel_center(tile['tile_cx'], tile['tile_cy'],
                                             tile_size, tile_size, 0, 0)
                z = tile['slice']
                record = {
                    'vx': int(px), 'vy': int(py), 'vz': int(z),
                    **{class_fraction_col(c): m for c, m in cov['class_means'].items()},
                    'he_coverage_frac': cov['he_frac'],
                }
                for name, frac in cov['mr_fracs'].items():
                    record[f'{name}_coverage_frac'] = frac
                for name, arr in cov['mr_subs'].items():
                    record[name] = nonzero_mean(arr)
                    for feat_name, val in radiomics_features(radiomics_extractor, arr).items():
                        record[f'{name}_rad_{feat_name}'] = val
                records.append(record)

            print(f"{block_name} (tile_size={tile_size}): {len(records):,}/{len(tiles):,} "
                  f"tile(s) kept (rest dropped for coverage)")
            if not records:
                continue

            merged = pd.DataFrame(records)
            merged = rename_to_standardized_mr_columns(merged,
                                                       rabbit_data[(rid, block_num)]['mr_labels'])
            out_path = os.path.join(block_analysis_dir(base_dir, rid, block_num),
                                    f'tile_features_by_id_size{tile_size}{csv_suffix}.csv')
            merged.to_csv(out_path, index=False)
            print(f"{block_name} (tile_size={tile_size}): saved {len(merged):,} row(s) -> {out_path}")


if __name__ == '__main__':
    BASE_DIR = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RENAME_DOC_PATH = '/Users/jbonaventura/Documents/MR_renames.numbers'
    RABBIT_BLOCKS = load_ready_rabbit_blocks()

    DAY = 'day3'
    # Day3End_Registered subfolder to read MR from: 'Normalized' (per-volume percentile),
    # 'Normalized_method2' (healthy-muscle ratio, fixed map scale), or '' (raw). Every output
    # CSV gets a matching suffix, so each setting keeps its own files.
    MR_NORM_DIR = 'Normalized_method2'
    # ProjectClassPredictionsToMR.py's LABEL_SOURCE: its per-class fraction maps are the targets.
    LABEL_SOURCE = 'R3_softmax'
    CSV_SUFFIX = f'{mr_norm_csv_suffix(MR_NORM_DIR)}_{LABEL_SOURCE}'
    TILE_SIZE = 224
    MIN_HE_FRACTION = 0.1
    MIN_MR_FRACTION = 0  # every contrast must individually clear this fraction
    # None = every contrast common to all rabbits. Voxels must be nonzero in EVERY listed
    # contrast, so fewer contrasts keeps more voxels (the T1/T2 maps have a narrow FOV).
    MR_CONTRASTS = None
    VOXEL_CSV_NAME = ('voxel_features_by_id' if MR_CONTRASTS is None else
                      'voxel_features_' + '_'.join(sorted(c.replace(' Day 3', '').replace(' ', '')
                                                          for c in MR_CONTRASTS))) + f'{CSV_SUFFIX}.csv'

    # Patch-preview settings only (not used by the real Stage 2 run). PATCH_SIZE must match
    # extract_mri_embedding.py's patch size (16) by hand.
    PATCH_SIZE = 16
    MIN_PATCH_HE_FRACTION = 0.0
    MIN_PATCH_MR_FRACTION = 0.0
    # Stage 2 sliding window: step (voxels) between patch grids, must divide the patch size.
    # None = the single unshifted grid -> patch_embeddings_by_id.csv.
    PATCH_STRIDE = 4
    # Stage 2 only: RadioDino needs [0, 1] input, but Normalized_method2's weighted contrasts are
    # ratios to healthy muscle (~1, tails to ~9). One fixed divisor per contrast for every rabbit
    # (near the pooled p99.9: T1w 3.0, T1w CE 3.8, T2w 4.8), clipped at 1. The T1/T2 maps are
    # already in [0, 1]. Must match MR_NORM_DIR; a contrast outside [0, 1] stops the run.
    EMBEDDING_DIVISORS = {'T1w Day 3': 4.0, 'T1w CE Day 3': 4.0, 'T2w Day 3': 5.0}

    # Stage 1b tile sizes: 2*radius+1 for Stage 1a radii 1/3/5, plus 16 = Stage 2's patch size.
    TILE1B_SIZES = [3, 7, 11, 16]
    MIN_TILE1B_HE_FRACTION = 0.5
    MIN_TILE1B_MR_FRACTION = 0.75

    # Per-run switches. Extraction stages overwrite their CSVs; leave off what you aren't rerunning.
    RUN_VOXELWISE = True  # Stage 0 + 1a
    RUN_TILEWISE = True   # Stage 2
    RUN_TILEWISE_AGGREGATE = True  # Stage 1b
    SHOW_PLOTS = False      # tile/coverage visualization, no saving
    SHOW_PATCH_PREVIEW = False  # patch-level coverage + content preview, no embedding run
    SHOW_GRADIENT_PREVIEW = False  # original vs. gradient_magnitude(), no tiling needed
    SHOW_HE_MR_SLICES = False  # H&E vs every loaded MR contrast, a few slices per rabbit
    SHOW_SLIDING_PATCHES = False  # H&E/MR slices with the Stage 2 patches kept at a few tile shifts

    rabbit_data = load_all_rabbit_data(RABBIT_BLOCKS, BASE_DIR, RENAME_DOC_PATH, LABEL_SOURCE,
                                        day=DAY, mr_norm_dir=MR_NORM_DIR,
                                        show_slices=SHOW_HE_MR_SLICES, contrasts=MR_CONTRASTS)
    if SHOW_HE_MR_SLICES:
        plt.show()

    if SHOW_SLIDING_PATCHES:
        for key, data in rabbit_data.items():
            # Diagonal shifts only: all PATCH_STRIDE shifts at once is an unreadable mesh.
            shown_shifts = [(d, d) for d in range(0, PATCH_SIZE, PATCH_STRIDE or PATCH_SIZE)]
            outlines, handles = sliding_patch_outlines(data, TILE_SIZE, PATCH_SIZE, shown_shifts,
                                                       min_he_fraction=0, min_mr_fraction=0)
            fig = plot_he_mr_slices(data['he_arr'], data['mr_volumes'], data['metadata'],
                                    labels=data['mr_labels'], margin=20, patch_outlines=outlines,
                                    title=f"{block_label(*key)}: Stage 2 patches kept at each tile shift")
            fig.legend(handles=handles, loc='outside right upper', fontsize=7)
        plt.show()

    if RUN_VOXELWISE:
        voxelwise_extraction(rabbit_data, BASE_DIR, extra_volumes_fn=combined_extra_volumes,
                              csv_name=VOXEL_CSV_NAME)

    # Stage 1b runs before Stage 2 (the slow one), so stopping Stage 2 early keeps every other
    # stage's CSVs.
    if RUN_TILEWISE_AGGREGATE:
        tilewise_aggregate_extraction(rabbit_data, BASE_DIR,
                                       tile_sizes=TILE1B_SIZES,
                                       min_he_fraction=MIN_TILE1B_HE_FRACTION,
                                       min_mr_fraction=MIN_TILE1B_MR_FRACTION,
                                       csv_suffix=CSV_SUFFIX)

    if RUN_TILEWISE:
        # Tile-level filters only skip tiles with no H&E/MR at all (tissue covers ~2% of a 224
        # tile); which patches are kept is decided per patch, after embedding.
        FMTilewiseExtraction(rabbit_data, BASE_DIR, tile_size=TILE_SIZE,
                              min_he_fraction=0, min_mr_fraction=0,
                              patch_stride=PATCH_STRIDE, csv_suffix=CSV_SUFFIX,
                              embedding_divisors=EMBEDDING_DIVISORS)

    if SHOW_PLOTS:
        all_tiles = build_all_rabbit_tiles(rabbit_data, TILE_SIZE, MIN_HE_FRACTION, MIN_MR_FRACTION)
        for key, tiles in all_tiles.items():
            name = block_label(*key)
            if not tiles:
                print(f"{name}: no tiles survived filtering, skipping plots")
                continue
            data = rabbit_data[key]
            plot_tile_bundles(tiles, labels=data['mr_labels'], n=5, title=name)
            z = pick_representative_slices(tiles, n=1)[0]
            plot_tile_grid_over_fov(data['he_arr'], data['mr_volumes'], tiles, z, TILE_SIZE,
                                     labels=data['mr_labels'], title=name)

            if SHOW_PATCH_PREVIEW:
                plot_patch_coverage_grid(tiles, PATCH_SIZE, MIN_PATCH_HE_FRACTION,
                                          MIN_PATCH_MR_FRACTION, labels=data['mr_labels'],
                                          n=5, title=name)
                plot_patch_samples(tiles, PATCH_SIZE, MIN_PATCH_HE_FRACTION,
                                    MIN_PATCH_MR_FRACTION, labels=data['mr_labels'],
                                    n=5, title=name)
            plt.show()

    if SHOW_GRADIENT_PREVIEW:
        for key in rabbit_data:
            plot_gradient_comparison(rabbit_data, key)
        plt.show()
