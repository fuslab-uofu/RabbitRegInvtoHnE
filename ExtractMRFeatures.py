import os
import math
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numbers_parser import Document

from PaintHnEFeatureToInVivo import read_axial_slice_metadata, valid_mask
from PullMRFeaturesByVoxel import load_mr_volumes, extract_voxelwise_features
from extract_mri_embedding import embed_tile, patch_pixel_center
from extract_mr_derived_features import gradient_magnitude, neighborhood_mean_std
from extract_radiomics_features import build_radiomics_extractor, radiomics_features

FEATURE_COL_DEFAULT = 'mean_Nucleus_Hematoxylin_OD_mean'


def classify_mr_column_day(column_name):
    """'day3' if the MR_renames.numbers header names it Day 3 (T1w/T2w/T1
    Map/T2 Map Day 3); everything else -- explicit 'Day 0' columns AND the
    untagged Adc Post/Pre, Max Temp Proj columns -- is day0 (confirmed with
    Jesse the untagged ones are Day 0 too)."""
    return 'day3' if 'Day 3' in column_name else 'day0'


def _read_rename_rows(rename_doc_path):
    """header row + {rabbit_id: row} from MR_renames.numbers, read once so
    callers that need every rabbit's row (columns_present_for_all_rabbits)
    don't re-parse the doc once per rabbit."""
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
    """Day-column names with a non-blank cell for EVERY rabbit in rabbit_ids --
    a doc-level pre-filter applied before any MR volume is loaded from disk,
    so a contrast that's blank for even one rabbit is never requested for any
    rabbit (rather than requested, loaded, and only dropped afterward by
    restrict_to_common_labels)."""
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
    """filename stem -> human-readable label, from MR_renames.numbers' header,
    for the given rabbit, restricted to columns matching `day` ('day3',
    'day0', or 'all') and, if given, further restricted to allowed_columns
    (see columns_present_for_all_rabbits). Blank cells (contrast not
    acquired/available for this rabbit) are skipped."""
    header, rows = _read_rename_rows(rename_doc_path)
    if rabbit_id not in rows:
        raise ValueError(f"No row for '{rabbit_id}' in {rename_doc_path}")
    rabbit_row = rows[rabbit_id]

    wanted = {col for col in header[1:]
              if (day == 'all' or classify_mr_column_day(col) == day)
              and (allowed_columns is None or col in allowed_columns)}
    return {val: col for col, val in zip(header, rabbit_row) if col in wanted and val}


def select_day_volumes(mr_volumes, stem_to_label):
    """Filter mr_volumes down to just the contrasts named in stem_to_label
    (matched by prefix against the rename doc's stems, checked in both
    directions -- the doc's value is sometimes a prefix of the loaded stem,
    and sometimes the reverse, per PullMRFeaturesByVoxel.py's own loading
    convention). Keeps the ORIGINAL stem as the dict key; returns a separate
    stem -> label lookup for cross-rabbit standardization."""
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


def restrict_to_common_labels(per_rabbit_labels):
    """{rabbit_id: {stem: label}} -> the set of labels present for every
    rabbit -- mirrors MultiRabAnalysis.py's feature_standardization, applied
    to MR contrast availability so every rabbit ends up contributing the same
    set of contrasts."""
    label_sets = {rid: set(labels.values()) for rid, labels in per_rabbit_labels.items()}
    common = set.intersection(*label_sets.values())
    varying = set.union(*label_sets.values()) - common
    print(f"{len(common)} MR contrast(s) common to all {len(per_rabbit_labels)} rabbit(s)")
    if varying:
        print(f"{len(varying)} contrast(s) vary across rabbits, dropped: {sorted(varying)}")
        for rid, names in label_sets.items():
            still_missing = common - names
            if still_missing:
                print(f"  Rabbit {rid} missing: {sorted(still_missing)}")
    return common


def filter_to_labels(selected, labels, keep_labels):
    return ({stem: arr for stem, arr in selected.items() if labels[stem] in keep_labels},
            {stem: lab for stem, lab in labels.items() if lab in keep_labels})


def rename_to_standardized_mr_columns(df, mr_labels):
    """Rename every output column that carries a raw MR contrast stem --
    either bare (Stage 0/1a/1b's own raw-intensity column) or as a prefix on
    a derived column ('{stem}_grad_axial', '{stem}_nbhd_mean_2d_r3',
    '{stem}_rad_glcm_Contrast', '{stem}_emb3', '{stem}_coverage_frac') -- to
    that contrast's standardized MR_renames label instead. Without this, two
    rabbits' CSVs name the same contrast differently (e.g. R24-240's
    '61_3d_vibe_05mmiso_cor_ce' vs R24-058's 'inVivoMR' for what
    MR_renames.numbers calls the same standardized contrast), which silently
    breaks any cross-rabbit pooling/analysis downstream. Longest-stem-first,
    and a column is claimed by at most one stem, so a shorter stem that
    happens to prefix a longer one's match can't steal it back."""
    rename_map = {}
    for stem in sorted(mr_labels, key=len, reverse=True):
        label = mr_labels[stem]
        for col in df.columns:
            if col in rename_map:
                continue
            if col == stem or col.startswith(stem + '_'):
                rename_map[col] = label + col[len(stem):]
    return df.rename(columns=rename_map)


def extract_tile(arr, cx, cy, tile_size):
    """Zero-padded tile_size x tile_size crop of a 2D (L-R, A-P) plane,
    centered at (cx, cy). Clamps the source read to the array's real bounds
    and computes the destination offset explicitly -- deliberately not a bare
    `arr[x0:x1, y0:y1]` slice, since that wraps around on negative indices
    instead of padding (the same bug the whole-slice coverage-fraction
    calculation sidestepped earlier by never windowing at all)."""
    half = tile_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + tile_size, y0 + tile_size

    tile = np.zeros((tile_size, tile_size), dtype=arr.dtype)
    src_x0, src_x1 = max(x0, 0), min(x1, arr.shape[0])
    src_y0, src_y1 = max(y0, 0), min(y1, arr.shape[1])
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    tile[dst_x0:dst_x0 + (src_x1 - src_x0), dst_y0:dst_y0 + (src_y1 - src_y0)] = \
        arr[src_x0:src_x1, src_y0:src_y1]
    return tile


def round_half_up(x):
    """Standard round-half-up, not Python's builtin round() (banker's
    rounding) -- so the grid-shape threshold at exactly 1.5x/2.5x/... behaves
    predictably instead of silently rounding to the nearest even integer."""
    return math.floor(x + 0.5)


def compute_grid_shape(extent, tile_size):
    """Tiles needed along one axis to cover a coverage extent: extent >=
    1.5*tile_size bumps from 1 to 2 tiles, >= 2.5*tile_size to 3, etc. n=1
    (extent comfortably fits in one tile) is the original single-big-tile
    case, not a separate code path -- see tile_centers."""
    return max(1, round_half_up(extent / tile_size))


def tile_centers(centroid, n, stride):
    """n evenly spaced tile centers along one axis, symmetric around
    centroid. n=1 reduces to exactly [centroid] -- the original single-tile
    behavior falls out of this formula rather than needing its own branch."""
    return [round_half_up(centroid + (i - (n - 1) / 2) * stride) for i in range(n)]


def build_tiles(he_arr, mr_volumes, metadata, tile_size, stride=None):
    """One tile bundle per (slice, row, col) -- grid shape per axis (L-R,
    A-P) derived from comparing that axis's H&E coverage extent (metadata's
    xlo/xhi, ylo/yhi) to tile_size, so a tile_size that already covers the
    whole extent collapses to one tile per slice at the centroid, same as
    before. `stride` defaults to tile_size (non-overlapping)."""
    if stride is None:
        stride = tile_size
    cx, cy = metadata['cx'], metadata['cy']
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
        for row, tx in enumerate(x_centers):
            for col, ty in enumerate(y_centers):
                he_tile = extract_tile(he_slice, tx, ty, tile_size)
                mr_tiles = {name: extract_tile(sl, tx, ty, tile_size)
                            for name, sl in mr_slices.items()}
                tiles.append({'slice': z, 'row': row, 'col': col,
                              'tile_cx': tx, 'tile_cy': ty,
                              'he_tile': he_tile, 'mr_tiles': mr_tiles})
    return tiles


def filter_tiles_by_he_coverage(tiles, min_fraction):
    """Drops tiles whose H&E coverage fraction (NaN-only valid_mask, same
    convention as PaintHnEFeatureToInVivo.py -- 0 is a real feature value,
    not missing data) falls below min_fraction, so downstream feature
    extraction/plots aren't full of tiles with little or no real H&E signal
    to serve as a ground-truth label."""
    kept = [t for t in tiles if valid_mask(t['he_tile']).mean() >= min_fraction]
    print(f"{len(kept)}/{len(tiles)} tile(s) kept at H&E coverage threshold {min_fraction}")
    return kept


def filter_tiles_by_mr_coverage(tiles, min_fraction=0.0):
    """Drops tiles where any single MR contrast's coverage fraction (share of
    nonzero pixels -- MR's own no-data convention, confirmed empirically in
    PullMRFeaturesByVoxel.py: 0.0 = background/no-data, no NaNs, unlike the
    NaN-only H&E volume) fails to clear min_fraction. EVERY contrast has to
    individually clear the bar, not just one of them -- min_fraction=0.0 (the
    default) means "must have at least one real pixel, not be entirely
    zero"; raise it for a real percentage bar, same idea as
    filter_tiles_by_he_coverage. Uses a strict > at the zero default (a
    fraction is never negative, so `>= 0.0` would silently keep even a fully
    zero contrast) and >= once a real threshold is set, matching the H&E
    filter's "less than X% is dropped" convention."""
    def contrast_ok(arr):
        frac = (arr != 0).mean()
        return frac > 0 if min_fraction <= 0 else frac >= min_fraction
    kept = [t for t in tiles if all(contrast_ok(arr) for arr in t['mr_tiles'].values())]
    print(f"{len(kept)}/{len(tiles)} tile(s) kept at MR coverage threshold "
          f"{min_fraction} (every contrast)")
    return kept


def run_feature_extraction(tiles, feature_extractor=None):
    """Calls feature_extractor(mr_tiles, he_tile) per tile bundle if given;
    otherwise passes the raw tiles through unchanged, so this is useful
    before any extractor is plugged in ("just save tiles and go from
    there")."""
    if feature_extractor is None:
        return tiles
    return [{'slice': t['slice'], 'features': feature_extractor(t['mr_tiles'], t['he_tile'])}
            for t in tiles]


def extract_mr_embeddings(tiles):
    """Per-(tile, patch) RadioDino embeddings -- flattens each tile's
    (14,14,768) grid to one row per patch, one MR contrast per embed_tile call
    (contrasts are never stacked as channels), but all tiles for a given
    contrast are batched into a single call (embed_tile chunks internally at
    its own batch_size) since RadioDino inference is far cheaper batched than
    one 224x224 tile at a time. Assumes every tile in `tiles` carries the same
    set of MR contrasts, true here since they all come from one rabbit's
    build_tiles call over one `selected` dict. Returns (rows, patch): `rows`
    carries `tile_i` (index back into `tiles`, so a caller can pull that
    tile's he_tile/mr_tiles/tile_cx/tile_cy/slice for coverage filtering,
    target aggregation, and voxel-ID lookup) plus the patch's own grid
    position; `patch` is the model's patch size in pixels (same for every
    contrast, one model for the whole call)."""
    if not tiles:
        return [], None
    contrast_names = list(tiles[0]['mr_tiles'].keys())
    embedded = {}
    for name in contrast_names:
        start = time.time()
        embedded[name] = embed_tile([t['mr_tiles'][name] for t in tiles])
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
    """n slice indices spread evenly across every slice that has tile(s) --
    picks unique slices, not raw list positions, since a slice can now hold
    more than one grid tile."""
    zs = sorted({t['slice'] for t in tiles})
    idx = np.linspace(0, len(zs) - 1, min(n, len(zs))).round().astype(int)
    return [zs[i] for i in idx]


def plot_tile_bundles(tiles, labels=None, n=5, title=None):
    """One row per representative slice, one column per MR contrast plus H&E --
    lets you eyeball tile content and zero-padding across every contrast side
    by side, the same idea as the old TileInVivoMR.py's per-contrast grid.
    Prefers the (row=0, col=0) tile at each chosen slice, falling back to
    whatever tile is actually present there if that one got dropped by
    coverage filtering -- a content spot check, not an enumeration of every
    grid cell (see plot_tile_grid_over_fov for the full grid). `labels`
    (stem -> display name) controls column titles only."""
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
    """Full (L-R, A-P) field of view for one slice, with every tile's
    footprint at that slice drawn as a rectangle -- one panel per MR contrast
    plus H&E. Shows where the grid actually sits relative to the whole FOV,
    complementing plot_tile_bundles' cropped-tile-content view."""
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


def plot_gradient_comparison(rabbit_data, rid, z=None):
    """Debug visualization: one column per MR contrast, original above its
    gradient_magnitude() below, at a single central slice -- computed fresh
    here (not from any saved extra_volumes) purely for eyeballing the
    gradient computation before it's wired into real extraction. Computes
    the gradient on just that one slice (arr[:, z:z+1, :], the same axis
    shape gradient_magnitude expects) rather than the whole volume, since
    only one slice is ever displayed. z defaults to the rabbit's own
    (zlo+zhi)//2 midpoint from its stored coverage metadata."""
    data = rabbit_data[rid]
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

    fig.suptitle(f"Rabbit {rid} -- original vs. gradient magnitude, slice {z}")
    fig.tight_layout()
    return fig


GRADIENT_AXIS_VARIANTS = [
    ('axial', (0, 2)),      # in-plane, matches this pipeline's per-slice default
    ('coronal', (0, 1)),
    ('sagittal', (1, 2)),
    ('3d', (0, 1, 2)),
]


def gradient_extra_volumes(data):
    """extra_volumes_fn adapter (see voxelwise_extraction) computing
    gradient_magnitude() for every MR contrast in data['mr_volumes'], across
    all 4 axis variants in GRADIENT_AXIS_VARIANTS -- cheap to compute all
    four since they're just different combinations of the same underlying
    np.gradient partials (see gradient_magnitude's own docstring). Column
    names: '{contrast}_grad_{axis_label}'."""
    extra = {}
    for name, arr in data['mr_volumes'].items():
        for label, axes in GRADIENT_AXIS_VARIANTS:
            extra[f'{name}_grad_{label}'] = gradient_magnitude(arr, axes=axes)
    return extra


# Confirmed v1 scope (see project memory): [1,3,5,7] voxels in 2D (in-plane),
# [1,3] in 3D -- 3D capped lower deliberately, roughly spans up to Stage 2's
# patch size. Raw MR contrasts only for now, not yet applied to gradient
# maps too (a later addition, same function either way).
NEIGHBORHOOD_RADII = {
    (0, 2): [1, 3, 5, 7],
    (0, 1, 2): [1, 3],
}
NEIGHBORHOOD_AXIS_LABELS = {(0, 2): '2d', (0, 1, 2): '3d'}


def neighborhood_extra_volumes(data):
    """extra_volumes_fn adapter (see voxelwise_extraction) computing
    neighborhood_mean_std() for every MR contrast in data['mr_volumes'], at
    every (axes, radius) combination in NEIGHBORHOOD_RADII. Column names:
    '{contrast}_nbhd_{mean|std}_{2d|3d}_r{radius}'."""
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
    """Merges every registered extra_volumes_fn adapter (gradient,
    neighborhood stats, ...) into the single callback voxelwise_extraction
    accepts."""
    extra = {}
    extra.update(gradient_extra_volumes(data))
    extra.update(neighborhood_extra_volumes(data))
    return extra


def load_rabbit_mr(rabbit_id, block_num, base_dir, rename_doc_path, day='day3',
                    normalized=True, allowed_columns=None, feature_col=FEATURE_COL_DEFAULT):
    Block = f'Block{block_num:02d}'
    rabbase = os.path.join(base_dir, f'R{rabbit_id}')
    analysis_dir = os.path.join(rabbase, 'Analysis', Block)
    he_path = os.path.join(analysis_dir, f'hne_{feature_col}_invivo.nii.gz')

    he_arr = nib.load(he_path).get_fdata(dtype=np.float32)
    metadata = read_axial_slice_metadata(he_path)

    voxel_maps_dir = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps')
    label_path = os.path.join(voxel_maps_dir, 'voxel_tile_vol.nii.gz')
    label_arr = nib.load(label_path).get_fdata().astype(np.int64)

    day3end_dir = os.path.join(rabbase, 'InVivo_MR/InVMRDataSets/Day3End_Registered')
    mr_dir = os.path.join(day3end_dir, 'Normalized') if normalized else day3end_dir
    mr_volumes = load_mr_volumes(mr_dir)

    stem_to_label = load_day_labels(rename_doc_path, f'R{rabbit_id}', day=day,
                                     allowed_columns=allowed_columns)
    selected, labels = select_day_volumes(mr_volumes, stem_to_label)
    print(f"Rabbit {rabbit_id}: {len(selected)}/{len(stem_to_label)} requested "
          f"{day} contrast(s) loaded")
    return he_arr, metadata, label_arr, selected, labels


def load_all_rabbit_data(rabbit_blocks, base_dir, rename_doc_path, day='day3',
                          normalized=True, feature_col=FEATURE_COL_DEFAULT):
    """Load + cross-rabbit-standardize every rabbit's H&E/voxel-ID-label/MR
    data once. Stage 0 (voxelwise), Stage 2 (tilewise), and interactive
    tile/coverage visualization all reuse this same load instead of each
    re-reading the same volumes from disk."""
    allowed_columns = columns_present_for_all_rabbits(
        rename_doc_path, [f'R{rid}' for rid in rabbit_blocks], day=day)

    per_rabbit = {rid: load_rabbit_mr(rid, block_num, base_dir, rename_doc_path, day,
                                       normalized, allowed_columns, feature_col)
                  for rid, block_num in rabbit_blocks.items()}

    common_labels = restrict_to_common_labels(
        {rid: labels for rid, (_, _, _, _, labels) in per_rabbit.items()})

    rabbit_data = {}
    for rid, (he_arr, metadata, label_arr, selected, labels) in per_rabbit.items():
        selected, labels = filter_to_labels(selected, labels, common_labels)
        rabbit_data[rid] = {
            'he_arr': he_arr, 'metadata': metadata, 'label_arr': label_arr,
            'mr_volumes': selected, 'mr_labels': labels,
        }
    return rabbit_data


def build_all_rabbit_tiles(rabbit_data, tile_size=224, min_he_fraction=None, min_mr_fraction=None):
    """Tile already-loaded rabbit data (see load_all_rabbit_data) at TILE_SIZE,
    applying the given coverage thresholds. Kept separate from loading so the
    same load can be tiled at multiple tile sizes without re-reading volumes."""
    all_tiles = {}
    for rid, data in rabbit_data.items():
        tiles = build_tiles(data['he_arr'], data['mr_volumes'], data['metadata'], tile_size)
        if min_he_fraction is not None:
            tiles = filter_tiles_by_he_coverage(tiles, min_he_fraction)
        if min_mr_fraction is not None:
            tiles = filter_tiles_by_mr_coverage(tiles, min_mr_fraction)
        all_tiles[rid] = tiles
    return all_tiles


def voxelwise_extraction(rabbit_data, rabbit_blocks, base_dir, feature_col=FEATURE_COL_DEFAULT,
                          extra_volumes_fn=None):
    """Stage 0: per-rabbit voxelwise MR-intensity features against
    feature_col, via PullMRFeaturesByVoxel's masking (real H&E and every MR
    contrast nonzero at that exact voxel) -- no tiling, one row per voxel.
    extra_volumes_fn, if given, is called per rabbit as extra_volumes_fn(data)
    (data being that rabbit's entry in rabbit_data) and must return a
    {name: volume} dict of derived feature maps on the same grid -- e.g.
    gradient magnitude, Frangi vesselness, neighborhood mean/std. These ride
    along as output columns but never affect the He/MR validity mask (see
    extract_voxelwise_features's mask_volumes vs extra_volumes). Not wired to
    anything yet -- this is just the extension point for when those derived
    features exist. Saves each rabbit's table next to its cellcount CSVs,
    matching PullMRFeaturesByVoxel.py's own output convention."""
    for rid, data in rabbit_data.items():
        extra_volumes = extra_volumes_fn(data) if extra_volumes_fn is not None else None
        merged = extract_voxelwise_features(data['he_arr'], data['label_arr'],
                                             data['mr_volumes'], feature_col,
                                             extra_volumes=extra_volumes)
        merged = rename_to_standardized_mr_columns(merged, data['mr_labels'])
        analysis_dir = os.path.join(base_dir, f'R{rid}', 'Analysis',
                                     f'Block{rabbit_blocks[rid]:02d}')
        out_path = os.path.join(analysis_dir, 'voxel_features_by_id.csv')
        merged.to_csv(out_path, index=False)
        print(f"Rabbit {rid}: saved {len(merged):,} row(s) -> {out_path}")


def _patch_coverage_ok(frac, min_fraction):
    """Same '>0 at the 0.0 default, >= once a real threshold is set'
    convention as filter_tiles_by_mr_coverage, just evaluated per patch."""
    return frac > 0 if min_fraction <= 0 else frac >= min_fraction


def _patch_slice(tile, patch, pi, pj):
    """(he_sub, mr_subs) -- patch (pi,pj)'s own sub-crop out of an
    already-cropped tile, at the given patch size."""
    r0, r1 = pi * patch, (pi + 1) * patch
    c0, c1 = pj * patch, (pj + 1) * patch
    he_sub = tile['he_tile'][r0:r1, c0:c1]
    mr_subs = {name: arr[r0:r1, c0:c1] for name, arr in tile['mr_tiles'].items()}
    return he_sub, mr_subs


def patch_coverage(tile, patch, pi, pj, min_patch_he_fraction, min_patch_mr_fraction):
    """He/MR sub-crops, per-channel coverage fractions, and pass/fail for
    tile's patch (pi,pj) at the given thresholds -- the single source of
    truth both for 'is this patch kept' AND for the coverage fractions saved
    alongside each kept patch (a patch that barely clears the threshold and
    one that's fully covered shouldn't be treated as equally trustworthy
    downstream), shared by FMTilewiseExtraction (applied after embedding) and
    the patch-preview plots below (applied before, so
    min_patch_he_fraction/min_patch_mr_fraction can be tuned without paying
    for the embedding step). Returns a dict rather than a tuple since there
    are now enough distinct pieces that positional unpacking stops being
    readable at call sites."""
    he_sub, mr_subs = _patch_slice(tile, patch, pi, pj)
    he_valid = valid_mask(he_sub)
    he_frac = float(he_valid.mean())
    mr_fracs = {name: float((arr != 0).mean()) for name, arr in mr_subs.items()}
    he_ok = _patch_coverage_ok(he_frac, min_patch_he_fraction)
    mr_ok = all(_patch_coverage_ok(f, min_patch_mr_fraction) for f in mr_fracs.values())
    return {
        'he_sub': he_sub, 'he_valid': he_valid, 'mr_subs': mr_subs,
        'he_frac': he_frac, 'mr_fracs': mr_fracs,
        'kept': he_ok and mr_ok,
    }


def _representative_tiles(tiles, n):
    """Same selection plot_tile_bundles uses: one tile (row=0,col=0, or
    whatever's there if that one got filtered) per representative slice."""
    zs = pick_representative_slices(tiles, n)
    bundles = []
    for z in zs:
        z_tiles = [t for t in tiles if t['slice'] == z]
        bundles.append(next((t for t in z_tiles if t['row'] == 0 and t['col'] == 0), z_tiles[0]))
    return bundles


def plot_patch_coverage_grid(tiles, patch, min_patch_he_fraction, min_patch_mr_fraction,
                              labels=None, n=5, title=None):
    """Representative tiles with patch boundaries drawn on top and a green
    overlay on patches that would survive FMTilewiseExtraction's
    post-embedding coverage filter at the given thresholds -- computed
    directly from he_tile/mr_tiles via patch_coverage, so
    min_patch_he_fraction/min_patch_mr_fraction can be tuned before paying
    for the (costly) embedding step."""
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
    """Random sample of N patches that would survive FMTilewiseExtraction's
    post-embedding coverage filter, from the same representative tiles
    plot_patch_coverage_grid draws -- shown the same way plot_tile_bundles
    shows whole tiles (one row per patch, one column per MR contrast plus
    H&E), so patch-level content can be spot-checked before paying for the
    embedding step."""
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


def FMTilewiseExtraction(rabbit_data, rabbit_blocks, base_dir, tile_size=224,
                          min_he_fraction=0.1, min_mr_fraction=0.1,
                          min_patch_he_fraction=0.0, min_patch_mr_fraction=0.0,
                          feature_col=FEATURE_COL_DEFAULT):
    """Stage 2: per-rabbit RadioDino patch embeddings against feature_col.
    Tiles at tile_size (min_he_fraction/min_mr_fraction gate which whole
    tiles get embedded at all), then flattens each tile's embedding grid to
    one row per 16x16 patch. Coverage filtering happens AFTER embedding
    extraction, not before -- the ViT's self-attention mixes information
    across the whole tile to produce each patch's embedding, so a patch can't
    be dropped pre-extraction without corrupting its neighbors' embeddings
    too (min_patch_he_fraction/min_patch_mr_fraction gate this second,
    per-patch pass). Each surviving patch is saved with its central voxel's
    ID (from the voxel-ID label volume, same grid as everything else here)
    and its feature_col ground truth (mean over the patch's own H&E
    sub-region)."""
    all_tiles = build_all_rabbit_tiles(rabbit_data, tile_size, min_he_fraction, min_mr_fraction)

    for rid, tiles in all_tiles.items():
        if not tiles:
            print(f"Rabbit {rid}: no tile(s) survived tile-level coverage filtering, skipping")
            continue

        n_contrasts = len(tiles[0]['mr_tiles'])
        print(f"Rabbit {rid}: embedding {len(tiles):,} tile(s) across {n_contrasts} contrast(s)...")
        rabbit_start = time.time()
        patch_rows, patch = extract_mr_embeddings(tiles)
        label_arr = rabbit_data[rid]['label_arr']

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
                'voxel_id': int(label_arr[px, z, py]),
                feature_col: float(cov['he_sub'][cov['he_valid']].mean()),
                # How much of this patch is real signal vs. NaN/zero background --
                # a barely-cleared-the-threshold patch and a fully-covered one both
                # count as "kept", but shouldn't be treated as equally trustworthy
                # downstream.
                'he_coverage_frac': cov['he_frac'],
            }
            for name, frac in cov['mr_fracs'].items():
                record[f'{name}_coverage_frac'] = frac
            for name, vec in row['features'].items():
                for d, val in enumerate(vec):
                    record[f'{name}_emb{d}'] = float(val)
            records.append(record)

        print(f"Rabbit {rid}: {len(records):,}/{len(patch_rows):,} patch(es) kept "
              f"(rest dropped for empty H&E/MR coverage)")
        if not records:
            continue

        merged = pd.DataFrame(records)
        merged = rename_to_standardized_mr_columns(merged, rabbit_data[rid]['mr_labels'])
        analysis_dir = os.path.join(base_dir, f'R{rid}', 'Analysis',
                                     f'Block{rabbit_blocks[rid]:02d}')
        out_path = os.path.join(analysis_dir, 'patch_embeddings_by_id.csv')
        merged.to_csv(out_path, index=False)
        print(f"Rabbit {rid}: saved {len(merged):,} row(s) -> {out_path} "
              f"(rabbit total: {time.time() - rabbit_start:.1f}s)")


def tilewise_aggregate_extraction(rabbit_data, rabbit_blocks, base_dir, tile_sizes,
                                   min_he_fraction, min_mr_fraction,
                                   feature_col=FEATURE_COL_DEFAULT):
    """Stage 1b: tile-level aggregation of BOTH the He target and every raw MR
    contrast (mean over the tile), at matching resolution to Stage 2 -- an
    apples-to-apples comparison Stage 1a's dense per-voxel granularity can't
    give on its own (a subsampled voxel is still one voxel's raw value, not a
    true block-averaged label).

    Reuses build_all_rabbit_tiles() for the grid and patch_coverage() for
    masked coverage/validity: passing the tile's own tile_size as the 'patch'
    size (pi=pj=0) degenerates patch_coverage() to describe the WHOLE tile,
    giving masked He/MR sub-crops and coverage fractions for free rather than
    duplicating that masking logic. Central voxel ID via patch_pixel_center()
    with the same patch==tile_size degenerate case -- exactly Stage 2's own
    convention, including its documented 1-pixel round-down for even sizes.

    One CSV per tile_size, not a single shared table: unlike Stage 1a's radii
    (which add columns to the SAME voxel rows), each tile_size here defines
    its own independent grid via build_tiles()'s own
    compute_grid_shape/tile_centers -- different tile_cx/tile_cy centers,
    different tile counts -- genuinely different row-sets, not the same rows
    described differently.

    Also runs PyRadiomics (firstorder/GLCM/GLRLM/GLSZM/GLDM/NGTDM) per tile
    per contrast, on the RAW MR contrast (not the derived gradient/
    neighborhood-stat maps -- see project discussion) masked by that
    contrast's own validity (same convention as everything else here). Not
    run on Stage 0/1a's per-voxel granularity -- PyRadiomics computes one
    feature vector per (image, mask) ROI, not a per-voxel sliding-window map,
    so it only fits tile-level extraction."""
    radiomics_extractor = build_radiomics_extractor()

    for tile_size in tile_sizes:
        all_tiles = build_all_rabbit_tiles(rabbit_data, tile_size, min_he_fraction, min_mr_fraction)

        for rid, tiles in all_tiles.items():
            if not tiles:
                print(f"Rabbit {rid} (tile_size={tile_size}): no tile(s) survived "
                      f"tile-level coverage filtering, skipping")
                continue

            label_arr = rabbit_data[rid]['label_arr']
            records = []
            for tile in tiles:
                cov = patch_coverage(tile, tile_size, 0, 0, min_he_fraction, min_mr_fraction)
                if not cov['kept']:
                    continue

                px, py = patch_pixel_center(tile['tile_cx'], tile['tile_cy'],
                                             tile_size, tile_size, 0, 0)
                z = tile['slice']
                record = {
                    'voxel_id': int(label_arr[px, z, py]),
                    feature_col: float(cov['he_sub'][cov['he_valid']].mean()),
                    'he_coverage_frac': cov['he_frac'],
                }
                for name, frac in cov['mr_fracs'].items():
                    record[f'{name}_coverage_frac'] = frac
                for name, arr in cov['mr_subs'].items():
                    valid = (arr != 0)
                    record[name] = float(arr[valid].mean()) if valid.any() else np.nan
                    for feat_name, val in radiomics_features(radiomics_extractor, arr).items():
                        record[f'{name}_rad_{feat_name}'] = val
                records.append(record)

            print(f"Rabbit {rid} (tile_size={tile_size}): {len(records):,}/{len(tiles):,} "
                  f"tile(s) kept (rest dropped for coverage)")
            if not records:
                continue

            merged = pd.DataFrame(records)
            merged = rename_to_standardized_mr_columns(merged, rabbit_data[rid]['mr_labels'])
            analysis_dir = os.path.join(base_dir, f'R{rid}', 'Analysis',
                                         f'Block{rabbit_blocks[rid]:02d}')
            out_path = os.path.join(analysis_dir, f'tile_features_by_id_size{tile_size}.csv')
            merged.to_csv(out_path, index=False)
            print(f"Rabbit {rid} (tile_size={tile_size}): saved {len(merged):,} row(s) -> {out_path}")


if __name__ == '__main__':
    BASE_DIR = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RENAME_DOC_PATH = '/Users/jbonaventura/Documents/MR_renames.numbers'
    RABBIT_BLOCKS = {"23-055": 7, "24-103": 6, "24-240": 3, "24-082": 5, "24-101": 11, "24-058": 6}

    DAY = 'day3'
    NORMALIZED = True
    FEATURE_COL = FEATURE_COL_DEFAULT
    TILE_SIZE = 224
    MIN_HE_FRACTION = 0.1
    MIN_MR_FRACTION = 0  # every contrast must individually clear this fraction

    # Patch-level preview only (see SHOW_PATCH_PREVIEW below) -- independent
    # of whatever thresholds FMTilewiseExtraction is actually called with, so
    # this can be tuned freely without touching the real run. PATCH_SIZE is
    # NOT derived from the model (that would mean loading it) -- must match
    # extract_mri_embedding.py's actual patch size (currently 16) by hand.
    PATCH_SIZE = 16
    MIN_PATCH_HE_FRACTION = 0.0
    MIN_PATCH_MR_FRACTION = 0.0

    # Stage 1b -- tile sizes chosen to roughly match Stage 1a's radii
    # (2*radius+1: 3,7,11 for radii 1,3,5) plus 16 to match Stage 2's own
    # patch granularity exactly; tune freely later, cheap to rerun.
    TILE1B_SIZES = [3, 7, 11, 16]
    MIN_TILE1B_HE_FRACTION = 0.5
    MIN_TILE1B_MR_FRACTION = 0.75

    # Independent per-run switches -- flip on whichever stage(s) you're
    # actually running; leave the rest off so a visualization/threshold-tuning
    # run doesn't also re-run (and overwrite) real extraction output.
    RUN_VOXELWISE = True  # Stage 0 + 1a
    RUN_TILEWISE = True   # Stage 2
    RUN_TILEWISE_AGGREGATE = False  # Stage 1b
    SHOW_PLOTS = False      # tile/coverage visualization, no saving
    SHOW_PATCH_PREVIEW = False  # patch-level coverage + content preview, no embedding run
    SHOW_GRADIENT_PREVIEW = False  # original vs. gradient_magnitude(), no tiling needed

    rabbit_data = load_all_rabbit_data(RABBIT_BLOCKS, BASE_DIR, RENAME_DOC_PATH, day=DAY,
                                        normalized=NORMALIZED, feature_col=FEATURE_COL)

    if RUN_VOXELWISE:
        voxelwise_extraction(rabbit_data, RABBIT_BLOCKS, BASE_DIR, feature_col=FEATURE_COL,
                              extra_volumes_fn=combined_extra_volumes)

    if RUN_TILEWISE:
        FMTilewiseExtraction(rabbit_data, RABBIT_BLOCKS, BASE_DIR, tile_size=TILE_SIZE,
                              min_he_fraction=0.05, min_mr_fraction=0,
                              feature_col=FEATURE_COL)

    if RUN_TILEWISE_AGGREGATE:
        tilewise_aggregate_extraction(rabbit_data, RABBIT_BLOCKS, BASE_DIR,
                                       tile_sizes=TILE1B_SIZES,
                                       min_he_fraction=MIN_TILE1B_HE_FRACTION,
                                       min_mr_fraction=MIN_TILE1B_MR_FRACTION,
                                       feature_col=FEATURE_COL)

    if SHOW_PLOTS:
        all_tiles = build_all_rabbit_tiles(rabbit_data, TILE_SIZE, MIN_HE_FRACTION, MIN_MR_FRACTION)
        for rid, tiles in all_tiles.items():
            if not tiles:
                print(f"Rabbit {rid}: no tiles survived filtering, skipping plots")
                continue
            data = rabbit_data[rid]
            plot_tile_bundles(tiles, labels=data['mr_labels'], n=5, title=f"Rabbit {rid}")
            z = pick_representative_slices(tiles, n=1)[0]
            plot_tile_grid_over_fov(data['he_arr'], data['mr_volumes'], tiles, z, TILE_SIZE,
                                     labels=data['mr_labels'], title=f"Rabbit {rid}")

            if SHOW_PATCH_PREVIEW:
                plot_patch_coverage_grid(tiles, PATCH_SIZE, MIN_PATCH_HE_FRACTION,
                                          MIN_PATCH_MR_FRACTION, labels=data['mr_labels'],
                                          n=5, title=f"Rabbit {rid}")
                plot_patch_samples(tiles, PATCH_SIZE, MIN_PATCH_HE_FRACTION,
                                    MIN_PATCH_MR_FRACTION, labels=data['mr_labels'],
                                    n=5, title=f"Rabbit {rid}")
            plt.show()

    if SHOW_GRADIENT_PREVIEW:
        for rid in rabbit_data:
            plot_gradient_comparison(rabbit_data, rid)
        plt.show()
