import os
import re
import glob
import warnings
from collections import namedtuple
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score
from SADMS import relaxation

Rabbit = namedtuple('Rabbit', ['rabbit_id', 'hne_names', 'mr_names', 'slides'])
Slide = namedtuple('Slide', ['basename', 'geometry', 'hne_features', 'mr_features'])

GEOMETRY_COLS = ['centroid_row', 'centroid_col']

# Standardized display ranges for relaxometry maps (ms), per ISMRM qMR study group
# recommendation (Fuderer et al. 2025) — same range used for all T1 maps and all T2
# maps so a given relaxation value always renders as the same color.
T1_RANGE = (0, 3000)
T2_RANGE = (0, 200)

# MR reconstruction bug (confirmed against raw data in Slicer, not a pipeline bug)
# occasionally produces scattered, physically-impossible relaxation values: negative
# T1/T2 (relaxation times are >= 0), and implausibly large T2. Voxels outside these
# bounds get NaN'd out in sanitize_relaxation_maps below.
T2_MAX_VALID = 250  # ms

# Fixed display range/colormap for % error maps. Fixing vmax=100 (rather than a
# data-driven percentile) means voxels with err=inf (y_true == 0) render as the
# top-of-range "over" color instead of collapsing the whole scale to ~0.
ERR_CMAP = 'magma_r'
ERR_VMIN, ERR_VMAX = 0, 100


def get_mr_style(variable_name):
    """Return a dict describing how to color a spatial map of this MR feature."""
    name = variable_name.lower()
    if 'temp' in name:
        return {'kind': 'cmap', 'cmap': 'hot'}
    if 'map' in name and 't1' in name:
        return {'kind': 'relaxation', 'maptype': 'T1', 'loLev': T1_RANGE[0], 'upLev': T1_RANGE[1]}
    if 'map' in name and 't2' in name:
        return {'kind': 'relaxation', 'maptype': 'T2', 'loLev': T2_RANGE[0], 'upLev': T2_RANGE[1]}
    return {'kind': 'cmap', 'cmap': 'gray'}


def load_slide(cellcount_path, data_dir):
    basename = os.path.basename(cellcount_path).replace("-cellcount.csv", "")
    texture_basename = basename.removesuffix("_corrected_fullres")
    texture_matches = glob.glob(os.path.join(data_dir, f"voxel_features_{texture_basename}*.csv"))
    if not texture_matches:
        print(f"No texture file found for {basename}, skipping.")
        return None
    cell_df = pd.read_csv(cellcount_path)
    texture_df = pd.read_csv(texture_matches[0])
    cell_df['poly_id'] = cell_df['tile_name'].str.extract(r'(\d+)$').astype(int)
    merged_df = texture_df.merge(cell_df, on='poly_id')

    haralick_cols = [c for c in texture_df.columns
                     if c.lower().startswith('h_') or c.lower().startswith('e_')]
    cell_cols = [c for c in cell_df.columns if c not in ('tile_name', 'poly_id')]
    mr_cols = [c for c in texture_df.columns
               if c not in ('poly_id', *GEOMETRY_COLS) and c not in haralick_cols]

    geometry = merged_df[GEOMETRY_COLS].to_numpy(dtype=float)
    hne_names = np.array(haralick_cols + cell_cols)
    hne_features = merged_df[haralick_cols + cell_cols].to_numpy(dtype=float)
    mr_names = np.array(mr_cols)
    mr_features = merged_df[mr_cols].to_numpy(dtype=float)

    return geometry, hne_features, hne_names, mr_features, mr_names, basename


def natural_sort_key(s):
    """Sort key that orders embedded numbers numerically (so '2a' < '10a'),
    unlike plain string sort where '10a' < '2a' lexicographically."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r'(\d+)', s)]


def select_cellcount_files(cellcount_files):
    """Only the full-res batch-pipeline's `_corrected_fullres` cellcount files --
    slides that haven't been re-run with full-res cell detection yet are
    excluded entirely, not filled in with the original lower-res counts."""
    fullres_files = [f for f in cellcount_files if f.endswith("_corrected_fullres-cellcount.csv")]
    fullres_slides = {os.path.basename(f).replace("-cellcount.csv", "").removesuffix("_corrected_fullres")
                       for f in fullres_files}
    all_slides = {os.path.basename(f).replace("-cellcount.csv", "").removesuffix("_corrected_fullres")
                  for f in cellcount_files}
    n_skipped = len(all_slides - fullres_slides)
    print(f"  {len(fullres_files)} slide(s) with full-res cell counts "
          f"({n_skipped} slide(s) skipped, no full-res version yet)")
    return fullres_files


def load_rabbit(rabbit_id, block_num):
    data_dir = os.path.join(BASE_DIR, f"R{rabbit_id}", "Analysis", f"Block{block_num:02d}")
    cellcount_files = glob.glob(os.path.join(data_dir, "HnE*cellcount.csv"))
    if not cellcount_files:
        print(f"No cellcount files found for rabbit {rabbit_id} in {data_dir}")
        return None
    cellcount_files = sorted(select_cellcount_files(cellcount_files), key=natural_sort_key)
    if not cellcount_files:
        print(f"No full-res cellcount files found for rabbit {rabbit_id} in {data_dir}")
        return None
    slides = []
    reference_hne_names = None
    reference_mr_names = None
    for f in cellcount_files:
        result = load_slide(f, data_dir)
        if result is None:
            continue
        geometry, hne_features, hne_names, mr_features, mr_names, basename = result
        if reference_hne_names is None:
            reference_hne_names = hne_names
            reference_mr_names = mr_names
        elif not np.array_equal(hne_names, reference_hne_names):
            print(f"Rabbit {rabbit_id}, slide {basename}: hne_names differ from "
                  f"rest of rabbit, skipping.")
            continue
        elif not np.array_equal(mr_names, reference_mr_names):
            print(f"Rabbit {rabbit_id}, slide {basename}: mr_names differ from "
                  f"rest of rabbit, skipping.")
            continue
        slides.append(Slide(basename=basename, geometry=geometry,
                             hne_features=hne_features, mr_features=mr_features))
    print(f"Rabbit {rabbit_id}: loaded {len(slides)} slides")
    if not slides:
        return None
    return Rabbit(rabbit_id=rabbit_id, hne_names=reference_hne_names, mr_names=reference_mr_names,
                  slides=slides)


def feature_standardization(per_rabbit_names):
    name_sets = {rabbit_id: set(names) for rabbit_id, names in per_rabbit_names.items()}
    common = set.intersection(*name_sets.values())
    union = set.union(*name_sets.values())
    varying = union - common

    print(f"{len(common)} features common to all {len(per_rabbit_names)} rabbits")
    print(f"{len(varying)} features vary across rabbits:")
    for rabbit_id, names in name_sets.items():
        missing = union - names
        if missing:
            print(f"  Rabbit {rabbit_id} missing: {sorted(missing)}")

    rabbit_ids = list(per_rabbit_names.keys())
    reference_id = rabbit_ids[0]
    reference_order = [n for n in per_rabbit_names[reference_id] if n in common]
    print(f"\nChecking common-feature ordering against rabbit {reference_id}:")
    for rabbit_id in rabbit_ids[1:]:
        this_order = [n for n in per_rabbit_names[rabbit_id] if n in common]
        if this_order != reference_order:
            print(f"  Rabbit {rabbit_id}: common feature order differs from rabbit {reference_id}")
        else:
            print(f"  Rabbit {rabbit_id}: common feature order matches rabbit {reference_id}")

    return common, varying


def load_mr_rename_table(path):
    """Return {rabbit_id: {raw_name: standardized_name}} from the wide rename CSV."""
    df = pd.read_csv(path).rename(columns={'Standardized': 'rabbit_id'})
    df['rabbit_id'] = df['rabbit_id'].str.lstrip('R')
    long_df = df.melt(id_vars='rabbit_id', var_name='standardized_name', value_name='raw_name')
    long_df = long_df.dropna(subset=['raw_name'])
    return {rabbit_id: dict(zip(g['raw_name'], g['standardized_name']))
            for rabbit_id, g in long_df.groupby('rabbit_id')}


def standardize_mr(rabbits, rename_csv_path):
    rename_table = load_mr_rename_table(rename_csv_path)
    wide = pd.read_csv(rename_csv_path).drop(columns=['Standardized'])
    keep_names = set(wide.columns[wide.notna().all(axis=0)])

    standardized_rabbits = []
    for rabbit in rabbits:
        raw_to_std = rename_table.get(rabbit.rabbit_id, {})
        keep_idx, new_mr_names = [], []
        for i, raw_name in enumerate(rabbit.mr_names):
            std_name = raw_to_std.get(raw_name)
            if std_name is None:
                print(f"Rabbit {rabbit.rabbit_id}: raw MR name '{raw_name}' not in rename table, dropping.")
                continue
            if std_name not in keep_names:
                continue
            keep_idx.append(i)
            new_mr_names.append(std_name)

        new_slides = [s._replace(mr_features=s.mr_features[:, keep_idx]) for s in rabbit.slides]
        standardized_rabbits.append(rabbit._replace(mr_names=np.array(new_mr_names), slides=new_slides))

    return standardized_rabbits


def sanitize_relaxation_maps(rabbits):
    """NaN out physically-impossible values in T1/T2 relaxation-map MR columns
    (see T2_MAX_VALID above for why). Relaxation columns are identified by name
    via get_mr_style, so this only ever touches actual T1/T2 maps, never
    T1w/T2w-weighted images or anything else in mr_names. NaN'ing (rather than
    dropping the voxel outright) means the voxel is excluded only from
    MR-involving analyses, via the isfinite checks every consumer already does
    (valid_voxel_mask, pca_by_rabbit, plot_feature_distributions_by_rabbit) —
    it's still usable for HnE-only analyses."""
    sanitized = []
    for rabbit in rabbits:
        styles = [get_mr_style(name) for name in rabbit.mr_names]
        new_slides = []
        for slide in rabbit.slides:
            mr_features = slide.mr_features.copy()
            for i, (name, style) in enumerate(zip(rabbit.mr_names, styles)):
                if style['kind'] != 'relaxation':
                    continue
                bad = mr_features[:, i] < 0
                if style['maptype'] == 'T2':
                    bad |= mr_features[:, i] > T2_MAX_VALID
                if bad.any():
                    print(f"Rabbit {rabbit.rabbit_id}, slide {slide.basename}: "
                          f"{bad.sum()} invalid value(s) in '{name}', setting to NaN")
                    mr_features[bad, i] = np.nan
            new_slides.append(slide._replace(mr_features=mr_features))
        sanitized.append(rabbit._replace(slides=new_slides))
    return sanitized


def valid_voxel_mask(mr_features, hne_features):
    """Rows (voxels) usable for cross-modal comparison: finite in both blocks
    (excludes NaN *and* +/-Inf — Haralick/texture ratio features can produce
    Inf on a degenerate/uniform patch, e.g. dividing by zero variance), and no
    zero-filled MR column (voxel fell outside that map's registered FOV --
    confirmed via FOVOVerLapHandaling.py that individual maps can be zero from
    FOV misalignment even when other maps in the same predictor set have real
    values, so any single zero column invalidates the voxel, not just an
    all-columns-zero row)."""
    valid = np.isfinite(np.hstack([mr_features, hne_features])).all(axis=1)
    valid &= ~(mr_features == 0).any(axis=1)
    return valid


def compute_cross_corr(mr_features, hne_features):
    valid = valid_voxel_mask(mr_features, hne_features)
    n_mr = mr_features.shape[1]
    corr, pval = spearmanr(np.hstack([mr_features[valid], hne_features[valid]]))
    return corr[:n_mr, n_mr:], pval[:n_mr, n_mr:]


def plot_hne_feature_spatial(rabbits, feature_name, cmap='viridis', vmin=None, vmax=None):
    # feat_idx looked up per-rabbit (not assumed to be the same column position
    # across rabbits) — load_rabbit already guarantees it's consistent within
    # a rabbit's own slides, and this way the plot doesn't depend on whether
    # hne_names order matches across rabbits.
    per_rabbit_panels = []
    for rabbit in rabbits:
        feat_idx = np.where(rabbit.hne_names == feature_name)[0]
        if len(feat_idx) == 0:
            raise ValueError(f"Feature '{feature_name}' not found in rabbit {rabbit.rabbit_id} hne_names")
        feat_idx = feat_idx[0]
        panels = [(slide.basename, slide.geometry, slide.hne_features[:, feat_idx])
                  for slide in rabbit.slides]
        per_rabbit_panels.append((rabbit.rabbit_id, panels))

    # Shared color scale across all rabbits' figures, unless the caller pins it
    # explicitly (e.g. vmin=0, vmax=1 for a feature with a known bounded range).
    if vmin is None or vmax is None:
        all_vals = np.concatenate([vals for _, panels in per_rabbit_panels for _, _, vals in panels])
        auto_vmin, auto_vmax = np.nanpercentile(all_vals, [1, 99])
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    for rabbit_id, panels in per_rabbit_panels:
        n_panels = len(panels)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False,
                                 layout='constrained')
        axes = axes.flatten()

        for ax, (basename, geometry, vals) in zip(axes, panels):
            valid = ~np.isnan(vals)
            sc = ax.scatter(geometry[valid, 0], geometry[valid, 1], c=vals[valid], s=8,
                            cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(basename, fontsize=9)
            ax.axis('off')

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f'Rabbit {rabbit_id} — {feature_name} at voxel locations', fontsize=12)
        fig.colorbar(sc, ax=axes[:n_panels].tolist(), label=feature_name, shrink=0.4)
        plt.show()


def plot_fold_spatial(fold_i, oof, geom, r2_scores, MR_names, slide_name):
    xy = geom[fold_i]
    ncols = 5
    nrows = int(np.ceil(len(MR_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for mr_j in range(len(MR_names)):
        y_true, y_pred = oof[mr_j][fold_i]
        ax = axes[mr_j]
        err = np.abs(y_pred - y_true) / np.abs(y_true) * 100
        vmax = np.nanpercentile(err, 95)
        sc  = ax.scatter(xy[:, 0], xy[:, 1], c=err, s=10, cmap='hot_r', vmin=0, vmax=vmax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(MR_names[mr_j], fontsize=8)
        ax.axis('off')
        plt.colorbar(sc, ax=ax, label='% error')
    for ax in axes[len(MR_names):]:
        ax.set_visible(False)
    fig.suptitle(f'Fold {fold_i + 1} — held out: {slide_name}', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_actual_pred_error_row(axes_row, xy, y_true, y_pred, row_title, style, value_range=None):
    """Actual / Predicted / % error scatter row for one (target, slide) pair,
    onto a pre-made row of 3 axes. Shared by plot_fold_comparison (rows=targets,
    pooled xy) and plot_fold_comparison_by_slide (rows=slides, per-slide xy)."""
    err = np.abs(y_pred - y_true) / np.abs(y_true) * 100
    if style['kind'] == 'relaxation':
        lo, hi = style['loLev'], style['upLev']
        true_vals, lut = relaxation(style['maptype'], y_true, lo, hi)
        pred_vals, _   = relaxation(style['maptype'], y_pred, lo, hi)
        value_cmap  = ListedColormap(lut)
        value_label = f"{style['maptype']} (ms)"
    else:
        if value_range is not None:
            lo, hi = value_range
        else:
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
        true_vals, pred_vals = y_true, y_pred
        value_cmap  = style['cmap']
        value_label = ''

    for col, (vals, title, cmap, vlo, vhi, clabel) in enumerate([
        (true_vals, 'Actual',    value_cmap, lo, hi,   value_label),
        (pred_vals, 'Predicted', value_cmap, lo, hi,   value_label),
        (err,       '% error',   ERR_CMAP,   ERR_VMIN, ERR_VMAX, '% error'),
    ]):
        ax = axes_row[col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=10,
                        cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'{row_title} — {title}' if col == 0 else title, fontsize=9)
        plt.colorbar(sc, ax=ax, label=clabel, fraction=0.046, pad=0.04)


def plot_fold_comparison(fold_i, oof, geom, target_names, fold_label, target_styles=None,
                         group_size=4, value_range=None):
    """Pooled view: all of a fold's test voxels on one scatter per target row."""
    xy = geom[fold_i]
    n_targets = len(target_names)

    for g_start in range(0, n_targets, group_size):
        group = list(range(g_start, min(g_start + group_size, n_targets)))
        fig, axes = plt.subplots(len(group), 3, figsize=(18, 4 * len(group)),
                                 squeeze=False)
        for row, target_j in enumerate(group):
            y_true, y_pred = oof[target_j][fold_i]
            style = target_styles[target_j] if target_styles is not None else {'kind': 'cmap', 'cmap': 'viridis'}
            plot_actual_pred_error_row(axes[row], xy, y_true, y_pred, target_names[target_j],
                                       style, value_range=value_range)

        fig.suptitle(f'Fold {fold_i + 1} — {fold_label}  '
                     f'(Target {g_start + 1}–{min(g_start + group_size, n_targets)} of {n_targets})',
                     fontsize=12)
        plt.tight_layout()
        plt.show()


def plot_fold_comparison_by_slide(fold_i, oof, geom, slide_ids, target_names, fold_label,
                                  target_styles=None, value_range=None, group_size=4):
    """Per-slide view: one row per slide in this fold's test set, instead of
    pooling every slide's voxels onto one overlaid scatter (which isn't
    spatially meaningful since centroid coordinates are local to each slide)."""
    fold_slide_ids = slide_ids[fold_i]
    slides_in_fold = sorted(set(fold_slide_ids))

    for target_j, target_name in enumerate(target_names):
        y_true, y_pred = oof[target_j][fold_i]
        style = target_styles[target_j] if target_styles is not None else {'kind': 'cmap', 'cmap': 'viridis'}

        for g_start in range(0, len(slides_in_fold), group_size):
            group = slides_in_fold[g_start:g_start + group_size]
            fig, axes = plt.subplots(len(group), 3, figsize=(18, 4 * len(group)), squeeze=False)
            for row, slide_name in enumerate(group):
                mask = fold_slide_ids == slide_name
                plot_actual_pred_error_row(axes[row], geom[fold_i][mask], y_true[mask], y_pred[mask],
                                           slide_name, style, value_range=value_range)

            fig.suptitle(f'Fold {fold_i + 1} — {fold_label} — {target_name}  '
                         f'(Slide {g_start + 1}–{min(g_start + group_size, len(slides_in_fold))} '
                         f'of {len(slides_in_fold)})', fontsize=12)
            plt.tight_layout()
            plt.show()


def named_col_idx(names, name, context):
    idx = np.where(names == name)[0]
    if len(idx) == 0:
        raise ValueError(f"'{name}' not found in {context}")
    return idx[0]


def pooled_hne_feature_range(rabbits, feature_name, lo_pct=1, hi_pct=99):
    """[lo_pct, hi_pct] percentile range for an HnE feature, pooled across every
    rabbit's slides. Reused as a shared vmin/vmax so the spatial map and the RF
    fold-comparison plots use the same color scale for a given feature."""
    per_rabbit_vals = []
    for rabbit in rabbits:
        feat_idx = named_col_idx(rabbit.hne_names, feature_name, f"rabbit {rabbit.rabbit_id} hne_names")
        per_rabbit_vals.append(np.concatenate([s.hne_features[:, feat_idx] for s in rabbit.slides]))
    return tuple(np.nanpercentile(np.concatenate(per_rabbit_vals), [lo_pct, hi_pct]))


def pool_rabbit_features(rabbit, predictor_names, target_names):
    """Stack a rabbit's slides together, selecting predictor columns from
    mr_features and target columns from hne_features by name (looked up per
    rabbit rather than assumed to share column order with other rabbits).
    Also returns a per-voxel slide_id array so pooled voxels can be split back
    out by slide later (e.g. for plotting)."""
    p_idx = [named_col_idx(rabbit.mr_names, n, f"rabbit {rabbit.rabbit_id} mr_names") for n in predictor_names]
    t_idx = [named_col_idx(rabbit.hne_names, n, f"rabbit {rabbit.rabbit_id} hne_names") for n in target_names]
    mr_selected  = np.vstack([s.mr_features for s in rabbit.slides])[:, p_idx]
    hne_selected = np.vstack([s.hne_features for s in rabbit.slides])[:, t_idx]
    geometry     = np.vstack([s.geometry for s in rabbit.slides])
    slide_ids    = np.concatenate([np.full(len(s.geometry), s.basename) for s in rabbit.slides])
    return mr_selected, hne_selected, geometry, slide_ids


def rabbit_holdout_folds(rabbits, predictor_names, target_names):
    """One fold per rabbit: train on all slides from the other rabbits pooled
    together, test on all slides from the held-out rabbit."""
    folds = []
    for fold_i, held_out in enumerate(rabbits):
        test_mr, test_hne, test_geom, test_slide_ids = pool_rabbit_features(held_out, predictor_names, target_names)
        train_parts = [pool_rabbit_features(r, predictor_names, target_names)
                       for j, r in enumerate(rabbits) if j != fold_i]
        train_mr  = np.vstack([p[0] for p in train_parts])
        train_hne = np.vstack([p[1] for p in train_parts])
        folds.append((f"Rabbit {held_out.rabbit_id}", train_mr, train_hne, test_mr, test_hne,
                      test_geom, test_slide_ids))
    return folds


def slide_holdout_folds(rabbit, predictor_names, target_names):
    """One fold per slide within a single rabbit: train on that rabbit's other
    slides, test on the held-out slide. For comparing a rabbit's own internal
    consistency against its cross-rabbit generalization (rabbit_holdout_folds)."""
    p_idx = [named_col_idx(rabbit.mr_names, n, f"rabbit {rabbit.rabbit_id} mr_names") for n in predictor_names]
    t_idx = [named_col_idx(rabbit.hne_names, n, f"rabbit {rabbit.rabbit_id} hne_names") for n in target_names]
    slide_mr   = [s.mr_features[:, p_idx] for s in rabbit.slides]
    slide_hne  = [s.hne_features[:, t_idx] for s in rabbit.slides]
    slide_geom = [s.geometry for s in rabbit.slides]

    folds = []
    for fold_i, slide in enumerate(rabbit.slides):
        other = [j for j in range(len(rabbit.slides)) if j != fold_i]
        train_mr  = np.vstack([slide_mr[j]  for j in other])
        train_hne = np.vstack([slide_hne[j] for j in other])
        test_slide_ids = np.full(len(slide_geom[fold_i]), slide.basename)
        folds.append((slide.basename, train_mr, train_hne, slide_mr[fold_i], slide_hne[fold_i],
                      slide_geom[fold_i], test_slide_ids))
    return folds


def run_rf_cv(folds, predictor_names, target_names, target_styles=None, plot_per_fold=True, value_range=None):
    """Run RF regression over prebuilt folds: each fold is
    (fold_label, train_mr, train_hne, test_mr, test_hne, test_geom, test_slide_ids).
    See rabbit_holdout_folds / slide_holdout_folds for fold construction."""
    n_folds      = len(folds)
    n_predictors = len(predictor_names)
    n_targets    = len(target_names)
    r2_scores   = np.full((n_folds, n_targets), np.nan)
    importances = np.zeros((n_folds, n_targets, n_predictors))
    # out-of-fold predictions: list of length n_targets, each a list of (y_true, y_pred) per fold
    oof  = [[[] for _ in range(n_folds)] for _ in range(n_targets)]
    geom = [None] * n_folds  # (x, y) centroid coords for each fold's test set
    slide_ids = [None] * n_folds  # per-voxel slide basename for each fold's test set
    fold_labels = [f[0] for f in folds]

    for fold_i, (fold_label, train_mr, train_hne, test_mr, test_hne, test_geom, test_slide_ids) in enumerate(folds):
        train_mask = valid_voxel_mask(train_mr, train_hne)
        test_mask  = valid_voxel_mask(test_mr, test_hne)
        X_train = train_mr[train_mask]
        X_test  = test_mr[test_mask]
        geom[fold_i] = test_geom[test_mask]
        slide_ids[fold_i] = test_slide_ids[test_mask]

        train_dropped_pct = 100 * (1 - train_mask.sum() / len(train_mask))
        test_dropped_pct  = 100 * (1 - test_mask.sum() / len(test_mask))
        print(f"Fold {fold_i + 1}/{n_folds} — held out: {fold_label}, "
              f"train={train_mask.sum()} voxels ({train_dropped_pct:.1f}% dropped), "
              f"test={test_mask.sum()} voxels ({test_dropped_pct:.1f}% dropped)")
        for target_j in range(n_targets):
            y_train = train_hne[train_mask, target_j]
            y_test  = test_hne[test_mask,   target_j]
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            r2_scores[fold_i, target_j]   = r2_score(y_test, y_pred)
            importances[fold_i, target_j] = rf.feature_importances_
            oof[target_j][fold_i] = (y_test, y_pred)
            print(f"  Target {target_j + 1}/{n_targets} done  R²={r2_scores[fold_i, target_j]:.3f}")

        if plot_per_fold:
            plot_fold_comparison_by_slide(fold_i, oof, geom, slide_ids, target_names, fold_label,
                                          target_styles=target_styles, value_range=value_range)
            plot_fold_importances(importances[fold_i:fold_i + 1], target_names, predictor_names, [fold_label])

    return r2_scores, importances, oof, geom, fold_labels, slide_ids
    # r2_scores:   (n_folds, n_targets)
    # importances: (n_folds, n_targets, n_predictors)
    # oof:         [n_targets][n_folds] -> (y_true, y_pred) arrays
    # geom:        [n_folds] -> (n_test_voxels, 2) centroid coords
    # fold_labels: [n_folds] -> held-out label (rabbit id or slide basename)
    # slide_ids:   [n_folds] -> (n_test_voxels,) slide basename per test voxel


def plot_r2(r2_scores, target_names, cv_label='leave-one-rabbit-out'):
    means = r2_scores.mean(axis=0)
    stds  = r2_scores.std(axis=0)
    x = np.arange(len(target_names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.7)
    for fold_r2 in r2_scores:
        ax.scatter(x, fold_r2, color='black', s=20, alpha=0.6, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title(f'RF regression: R² per target feature ({cv_label})')
    plt.tight_layout()
    plt.show()


def plot_best_predicted_vs_actual(r2_scores, oof, target_names, fold_labels):
    best_target = int(r2_scores.mean(axis=0).argmax())
    mean_r2 = r2_scores[:, best_target].mean()
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_labels)))

    fig, ax = plt.subplots(figsize=(7, 7))
    all_vals = []
    for fold_i, (y_true, y_pred) in enumerate(oof[best_target]):
        ax.scatter(y_true, y_pred, s=4, alpha=0.4, color=colors[fold_i], label=fold_labels[fold_i])
        all_vals.extend([y_true.min(), y_true.max()])

    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel(f'Actual {target_names[best_target]}')
    ax.set_ylabel(f'Predicted {target_names[best_target]}')
    ax.set_title(f'Predicted vs actual — {target_names[best_target]}  (mean R²={mean_r2:.3f})')
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_spatial_accuracy(r2_scores, oof, geom, target_names, fold_labels):
    best_target = int(r2_scores.mean(axis=0).argmax())
    n_folds = len(fold_labels)
    ncols = 3
    nrows = int(np.ceil(n_folds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for fold_i, (y_true, y_pred) in enumerate(oof[best_target]):
        ax = axes[fold_i]
        xy  = geom[fold_i]
        err = np.abs(y_pred - y_true) / np.abs(y_true) * 100
        sc  = ax.scatter(xy[:, 0], xy[:, 1], c=err, s=15, cmap=ERR_CMAP, vmin=ERR_VMIN, vmax=ERR_VMAX)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(fold_labels[fold_i], fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(sc, ax=ax, label='% error')

    for ax in axes[n_folds:]:
        ax.set_visible(False)

    fig.suptitle(f'Spatial prediction error — {target_names[best_target]}', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_importances(importances, target_names, predictor_names, top_n=10):
    mean_imp = importances.mean(axis=0)  # (n_targets, n_predictors)
    top_n = min(top_n, importances.shape[-1])
    n_targets = len(target_names)
    ncols = 4
    nrows = int(np.ceil(n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout='constrained')
    axes = np.array(axes).flatten()
    for target_j in range(n_targets):
        imp = mean_imp[target_j]
        top_idx = np.argsort(imp)[::-1][:top_n]
        ax = axes[target_j]
        ax.barh(range(top_n), imp[top_idx[::-1]], color='steelblue', alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([predictor_names[i] for i in top_idx[::-1]], fontsize=7)
        ax.set_title(target_names[target_j], fontsize=8)
        ax.set_xlabel('Importance', fontsize=7)
    for ax in axes[n_targets:]:
        ax.set_visible(False)
    fig.suptitle(f'RF feature importances — mean across all folds\nTop {top_n} predictor features per target', fontsize=11)
    plt.show()


def plot_fold_importances(importances, target_names, predictor_names, fold_labels, top_n=10):
    n_folds, n_targets, n_predictors = importances.shape
    top_n = min(top_n, n_predictors)
    ncols = 4
    nrows = int(np.ceil(n_targets / ncols))
    for fold_i in range(n_folds):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout='constrained')
        axes = np.array(axes).flatten()
        for target_j in range(n_targets):
            imp = importances[fold_i, target_j]
            top_idx = np.argsort(imp)[::-1][:top_n]
            ax = axes[target_j]
            ax.barh(range(top_n), imp[top_idx[::-1]], color='steelblue', alpha=0.7)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([predictor_names[i] for i in top_idx[::-1]], fontsize=7)
            ax.set_title(target_names[target_j], fontsize=8)
            ax.set_xlabel('Importance', fontsize=7)
        for ax in axes[n_targets:]:
            ax.set_visible(False)
        fig.suptitle(f'Fold {fold_i + 1} — held out: {fold_labels[fold_i]}\n'
                     f'Top {top_n} predictor features per target', fontsize=11)
        plt.show()


def plot_hne_variance(HnE_all, HnE_names, valid):
    data = HnE_all[valid]
    cv = np.nanstd(data, axis=0) / np.abs(np.nanmean(data, axis=0))
    x = np.arange(len(HnE_names))
    fig, ax = plt.subplots(figsize=(max(12, len(HnE_names) * 0.4), 5))
    ax.bar(x, cv, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(HnE_names, rotation=90, ha='right', fontsize=7)
    ax.set_ylabel('Coefficient of variation (σ / |μ|)')
    ax.set_title('HnE feature coefficient of variation (pooled across all slides)')
    plt.tight_layout()
    plt.show()


def plot_corr(ax, cross_corr, MR_names, HnE_names, title):
    im = ax.imshow(cross_corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(HnE_names)))
    ax.set_xticklabels(HnE_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(MR_names)))
    ax.set_yticklabels(MR_names, fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


def pooled_hne_names(rabbits):
    """hne_names, confirmed identical in order across all rabbits (verified
    explicitly, since pooling with mismatched column order would silently
    corrupt any pooled analysis rather than error out)."""
    reference = rabbits[0].hne_names
    for rabbit in rabbits[1:]:
        if not np.array_equal(rabbit.hne_names, reference):
            raise ValueError(
                f"Rabbit {rabbit.rabbit_id}: hne_names order differs from "
                f"rabbit {rabbits[0].rabbit_id}, cannot pool across rabbits."
            )
    return reference


def pca_by_rabbit(rabbits, feature_kind, n_components=2, n_top_loadings=15):
    """PCA on z-scored features pooled across all rabbits (either 'hne' or
    'mr'), colored by rabbit ID — batch-effect diagnostic: do voxels separate
    by rabbit? Z-scoring uses the pooled mean/std (not per-rabbit), since
    per-rabbit standardization would erase the very effect being checked for.
    Silhouette score is computed on the full standardized feature space, not
    the 2D PCA projection, since PCA distances in just 2 components aren't a
    reliable stand-in for separation in the full feature space."""
    if feature_kind == 'hne':
        feature_names = pooled_hne_names(rabbits)
        get_features = lambda s: s.hne_features
    elif feature_kind == 'mr':
        feature_names = rabbits[0].mr_names  # order already guaranteed by standardize_mr
        get_features = lambda s: s.mr_features
    else:
        raise ValueError(f"Unknown feature_kind: {feature_kind!r}")

    features = np.vstack([get_features(s) for r in rabbits for s in r.slides])
    rabbit_ids = np.concatenate([np.full(len(s.geometry), r.rabbit_id)
                                 for r in rabbits for s in r.slides])

    valid = np.isfinite(features).all(axis=1)  # excludes NaN and +/-Inf
    if feature_kind == 'mr':
        valid &= ~(features == 0).all(axis=1)
    print(f"{feature_kind.upper()}: {valid.sum()}/{len(valid)} voxels valid, "
          f"dropping {(~valid).sum()} (non-finite feature values"
          + (" or all-zero MR" if feature_kind == 'mr' else "") + ")")
    features, rabbit_ids = features[valid], rabbit_ids[valid]

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (features - features.mean(axis=0)) / features.std(axis=0)

    # Drop any feature with a degenerate z-score — either non-finite (exactly-
    # zero variance giving 0/0), or finite but absurdly large (variance so
    # close to zero, though not exactly 0, that a handful of z-scores blow up
    # to e.g. 1e200 — still "finite" by isfinite's definition, but PCA's
    # covariance matmul squares every value, and no real feature should ever
    # be >1e6 standard deviations from its mean, so this is unambiguously a
    # numerical artifact rather than a real outlier).
    Z_MAGNITUDE_LIMIT = 1e6
    bad_cols = ~np.isfinite(z).all(axis=0) | (np.abs(z) > Z_MAGNITUDE_LIMIT).any(axis=0)
    if bad_cols.any():
        print(f"Dropping {bad_cols.sum()} {feature_kind.upper()} feature(s) with degenerate "
              f"z-scores (near-zero variance and/or extreme outlier values): "
              f"{list(np.asarray(feature_names)[bad_cols])}")
        z = z[:, ~bad_cols]
        feature_names = np.asarray(feature_names)[~bad_cols]

    # numpy built against Apple's Accelerate BLAS/LAPACK emits spurious
    # divide-by-zero/overflow/invalid-value RuntimeWarnings from matmul-heavy
    # operations (PCA's covariance step, silhouette_score's pairwise distances)
    # even when every input and output value is finite and correct — verified
    # directly (z is fully finite here; PCA's output remains fully finite
    # despite the warning). Known numpy+Accelerate quirk on macOS, not a real
    # numerical problem, so suppressed rather than left to alarm on every run.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(z)
        sil = silhouette_score(z, rabbit_ids)

    fig, ax = plt.subplots(figsize=(8, 7))
    for rabbit_id in sorted(set(rabbit_ids)):
        mask = rabbit_ids == rabbit_id
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5, label=f'Rabbit {rabbit_id}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_title(f'{feature_kind.upper()} features PCA, colored by rabbit  (silhouette={sil:.3f})')
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"\n--- PCA on {feature_kind.upper()} features ---")
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1] * 100:.1f}% "
          f"(cumulative {pca.explained_variance_ratio_[:2].sum() * 100:.1f}% of total variance)")
    print(f"Silhouette score (rabbit ID as group label, full {z.shape[1]}-D feature space): {sil:.3f}")
    for pc_i in range(n_components):
        loadings = pca.components_[pc_i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:n_top_loadings]
        print(f"Top PC{pc_i + 1} loadings:")
        for i in top_idx:
            print(f"  {feature_names[i]}: {loadings[i]:+.3f}")

    return coords, rabbit_ids, pca


def plot_feature_distributions_by_rabbit(rabbits, feature_names, feature_kind, group_size=6,
                                         jitter_width=0.15, point_size=4, point_alpha=0.05,
                                         violin_alpha=0.6):
    """Violin plot per feature, one violin per rabbit, of that feature's pooled
    voxel-value distribution (all slides in the rabbit stacked together) — a
    direct look at whether rabbits differ in scale/spread, ahead of any
    modeling. feature_kind picks which array/name-list to pull from ('hne' or
    'mr'). Every voxel is filtered by valid_voxel_mask over that rabbit's full
    mr_features/hne_features (not just the one feature being plotted): finite
    in both MR and HnE, but the exact-zero check only applies to MR (HnE's
    own zero handling is a separate question, not yet addressed) — the same
    rule run_rf_cv and compute_cross_corr use, so these plots show exactly
    the voxel population the models actually see, not a looser per-feature
    filter.

    Every surviving voxel is also scattered next to its violin with
    horizontal jitter (fixed seed for reproducibility), instead of
    matplotlib's default min/max whisker line — for skewed features the
    whisker can shoot out to a single extreme value with no sense of how many
    points are actually out there; the scatter shows the real outlier
    density."""
    if feature_kind == 'hne':
        get_names, get_features = lambda r: r.hne_names, lambda s: s.hne_features
    elif feature_kind == 'mr':
        get_names, get_features = lambda r: r.mr_names, lambda s: s.mr_features
    else:
        raise ValueError(f"Unknown feature_kind: {feature_kind!r}")

    rabbit_ids = [r.rabbit_id for r in rabbits]
    ncols = min(3, group_size)
    rng = np.random.default_rng(42)

    rabbit_valid_masks = []
    for rabbit in rabbits:
        mr_pooled  = np.vstack([s.mr_features for s in rabbit.slides])
        hne_pooled = np.vstack([s.hne_features for s in rabbit.slides])
        rabbit_valid_masks.append(valid_voxel_mask(mr_pooled, hne_pooled))

    for g_start in range(0, len(feature_names), group_size):
        group = feature_names[g_start:g_start + group_size]
        nrows = int(np.ceil(len(group) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False,
                                 layout='constrained')
        axes = axes.flatten()

        for ax, feature_name in zip(axes, group):
            per_rabbit_vals = []
            for rabbit, valid in zip(rabbits, rabbit_valid_masks):
                feat_idx = named_col_idx(get_names(rabbit), feature_name,
                                          f"rabbit {rabbit.rabbit_id} {feature_kind}_names")
                vals = np.concatenate([get_features(s)[:, feat_idx] for s in rabbit.slides])
                per_rabbit_vals.append(vals[valid])

            for i, vals in enumerate(per_rabbit_vals):
                jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals))
                ax.scatter(i + jitter, vals, s=point_size, alpha=point_alpha, color='black',
                          linewidths=0, zorder=1)

            violin = ax.violinplot(per_rabbit_vals, positions=range(len(rabbits)), showmedians=True,
                                   showextrema=False)
            for body in violin['bodies']:
                body.set_alpha(violin_alpha)
                body.set_zorder(2)
            violin['cmedians'].set_zorder(2)
            ax.set_xticks(range(len(rabbits)))
            ax.set_xticklabels(rabbit_ids, fontsize=8)
            ax.set_title(feature_name, fontsize=9)

        for ax in axes[len(group):]:
            ax.set_visible(False)

        fig.suptitle(f'{feature_kind.upper()} feature distributions by rabbit '
                     f'(Feature {g_start + 1}–{min(g_start + group_size, len(feature_names))} '
                     f'of {len(feature_names)})', fontsize=12)
        plt.show()


# --- Load all slides ---

BASE_DIR = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/"
RABBIT_BLOCKS = {"23-055": 7, "24-103": 6, "24-240": 3, "24-082": 5}
Rename_Doc = "/Users/jbonaventura/Documents/MR_renames.csv"
HNE_FEATURE_OF_INTEREST = "mean_Nucleus_Hematoxylin_OD_mean"  # column from hne_names — spatial map + RF target
# Which MR features to use as RF predictors: "day3" (only standardized names
# containing "Day 3"), "day0" (only "Day 0"), or "all" (every universal MR
# feature, Day 3 + Day 0 + Max Temp Proj mixed together).
MR_PREDICTOR_SET = "day3"

# Feature names to plot in the by-rabbit distribution violins (RUN_HNE/MR_FEATURE_
# DISTRIBUTIONS below) — None means "every hne_names/mr_names feature".
HNE_DISTRIBUTION_FEATURES = None
MR_DISTRIBUTION_FEATURES = None

# --- Toggle which analyses run below (setup/shared values above each guarded ---
# --- block always run, since later blocks may depend on them) ---
RUN_PCA_BATCH_CHECK           = False
RUN_HNE_FEATURE_DISTRIBUTIONS = False
RUN_MR_FEATURE_DISTRIBUTIONS  = True
RUN_SPATIAL_MAP               = True
RUN_CROSS_CORR                = False
RUN_RF_RABBIT_HOLDOUT         = True
RUN_RF_SLIDE_HOLDOUT          = True

all_rabbits = []
for rabbit_id, block_num in RABBIT_BLOCKS.items():
    rabbit = load_rabbit(rabbit_id, block_num)
    if rabbit is not None:
        all_rabbits.append(rabbit)


print("--- HnE feature standardization ---")
feature_standardization({r.rabbit_id: list(r.hne_names) for r in all_rabbits})

print("\n--- MR feature standardization (raw names) ---")
feature_standardization({r.rabbit_id: list(r.mr_names) for r in all_rabbits})

all_rabbits = standardize_mr(all_rabbits, Rename_Doc)

print("\n--- MR feature standardization (standardized names) ---")
feature_standardization({r.rabbit_id: list(r.mr_names) for r in all_rabbits})

print("\n--- Sanitizing relaxation maps (impossible T1/T2 values -> NaN) ---")
all_rabbits = sanitize_relaxation_maps(all_rabbits)


# --- Examples: working with the loaded Rabbit/Slide namedtuples ---

# Which rabbits loaded successfully
print("Rabbits loaded:", [r.rabbit_id for r in all_rabbits])

# Slide basenames and feature names for one rabbit
first_rabbit = all_rabbits[0]
print(f"Rabbit {first_rabbit.rabbit_id} slides:", [s.basename for s in first_rabbit.slides])
print(f"Rabbit {first_rabbit.rabbit_id} hne_names:", first_rabbit.hne_names)
print(f"Rabbit {first_rabbit.rabbit_id} mr_names:", first_rabbit.mr_names)

# Loop over every rabbit and every slide
for rabbit in all_rabbits:
    print(f"\nRabbit {rabbit.rabbit_id}: {len(rabbit.slides)} slides, "
          f"{len(rabbit.hne_names)} HnE features, {len(rabbit.mr_names)} MR features")
    for slide in rabbit.slides:
        print(f"  {slide.basename}: geometry shape {slide.geometry.shape}, "
              f"hne_features shape {slide.hne_features.shape}, "
              f"mr_features shape {slide.mr_features.shape}")

# --- Batch-effect check: do voxels separate by rabbit? PCA + silhouette score, ---
# --- run separately for HnE and MR features so we can see which modality (if  ---
# --- either) is driving any separation. Run first, before the other plots.    ---
reference_hne_names = pooled_hne_names(all_rabbits)

if RUN_PCA_BATCH_CHECK:
    pca_by_rabbit(all_rabbits, 'hne')
    pca_by_rabbit(all_rabbits, 'mr')

# --- Feature distributions by rabbit: one violin per rabbit per feature ---
if RUN_HNE_FEATURE_DISTRIBUTIONS:
    hne_dist_features = HNE_DISTRIBUTION_FEATURES if HNE_DISTRIBUTION_FEATURES is not None else list(reference_hne_names)
    plot_feature_distributions_by_rabbit(all_rabbits, hne_dist_features, 'hne')

if RUN_MR_FEATURE_DISTRIBUTIONS:
    mr_dist_features = MR_DISTRIBUTION_FEATURES if MR_DISTRIBUTION_FEATURES is not None else list(all_rabbits[0].mr_names)
    plot_feature_distributions_by_rabbit(all_rabbits, mr_dist_features, 'mr')

# --- Spatial map of a single HnE feature, every slide from every rabbit ---
# One big grid across all rabbits, just to eyeball how standardized this feature
# looks slide-to-slide/rabbit-to-rabbit before we design anything smarter.
# Fixed color range (reused below for the RF fold-comparison plots too, so
# Actual/Predicted panels share the same scale across every figure).
target_value_range = pooled_hne_feature_range(all_rabbits, HNE_FEATURE_OF_INTEREST)
if RUN_SPATIAL_MAP:
    plot_hne_feature_spatial(all_rabbits, HNE_FEATURE_OF_INTEREST, vmin=target_value_range[0], vmax=target_value_range[1])

if RUN_CROSS_CORR:
    # --- Pooled correlation figure (per rabbit) ---
    for rabbit in all_rabbits:
        mr_pooled  = np.vstack([s.mr_features for s in rabbit.slides])
        hne_pooled = np.vstack([s.hne_features for s in rabbit.slides])
        cross_corr_pooled, _ = compute_cross_corr(mr_pooled, hne_pooled)

        fig, ax = plt.subplots(figsize=(18, 5))
        im = plot_corr(ax, cross_corr_pooled, rabbit.mr_names, rabbit.hne_names,
                        f"Rabbit {rabbit.rabbit_id} — all slides pooled")
        plt.colorbar(im, ax=ax, label='Spearman r')
        plt.tight_layout()
        plt.show()

    # --- Pooled correlation figure (all rabbits) ---
    # reference_hne_names already verified (order matches across all rabbits) above,
    # by pooled_hne_names(), before the PCA batch-effect check.
    mr_pooled  = np.vstack([s.mr_features for r in all_rabbits for s in r.slides])
    hne_pooled = np.vstack([s.hne_features for r in all_rabbits for s in r.slides])
    cross_corr_pooled, _ = compute_cross_corr(mr_pooled, hne_pooled)

    fig, ax = plt.subplots(figsize=(18, 5))
    im = plot_corr(ax, cross_corr_pooled, all_rabbits[0].mr_names, reference_hne_names,
                    "All rabbits, all slides pooled")
    plt.colorbar(im, ax=ax, label='Spearman r')
    plt.tight_layout()
    plt.show()


# --- Random forest regression (leave-one-rabbit-out, MR -> single HnE feature) ---
# Predictor columns come from mr_names (consistent order across rabbits, guaranteed
# by standardize_mr); direction is hardcoded MR -> HnE for now since that's what
# we're testing first. Swap to HnE -> MR later by passing hne_names as predictors
# and a list of mr_names as targets instead — run_rf_loocv itself is direction-
# agnostic (pool_rabbit_features just assumes predictor <- mr_features, target <- hne_features).
predictor_names = all_rabbits[0].mr_names
if MR_PREDICTOR_SET == "day3":
    predictor_names = predictor_names[np.array(['Day 3' in n for n in predictor_names])]
elif MR_PREDICTOR_SET == "day0":
    predictor_names = predictor_names[np.array(['Day 0' in n for n in predictor_names])]
elif MR_PREDICTOR_SET != "all":
    raise ValueError(f"Unknown MR_PREDICTOR_SET: {MR_PREDICTOR_SET!r}, expected 'day3'/'day0'/'all'")
target_names = np.array([HNE_FEATURE_OF_INTEREST])
print(f"RF predictors (MR_PREDICTOR_SET={MR_PREDICTOR_SET!r}): {len(predictor_names)} MR features: {list(predictor_names)}")

# --- Leave-one-rabbit-out: cross-rabbit generalization ---
if RUN_RF_RABBIT_HOLDOUT:
    r2_scores, importances, oof, geom, fold_labels, slide_ids = run_rf_cv(
        rabbit_holdout_folds(all_rabbits, predictor_names, target_names), predictor_names, target_names,
        value_range=target_value_range)

    plot_r2(r2_scores, target_names)
    plot_best_predicted_vs_actual(r2_scores, oof, target_names, fold_labels)
    plot_spatial_accuracy(r2_scores, oof, geom, target_names, fold_labels)
    plot_importances(importances, target_names, predictor_names)

# --- Leave-one-slide-out within each rabbit: baseline internal consistency, ---
# --- for comparison against the cross-rabbit result above. One representative ---
# --- fold (central slide, by natural-sorted position) gets the same           ---
# --- Actual/Predicted/%error figure as each rabbit's per-slide row in the     ---
# --- cross-rabbit run above — rather than that figure repeated for every      ---
# --- slide in the rabbit. Same fixed color range.                             ---
if RUN_RF_SLIDE_HOLDOUT:
    for rabbit in all_rabbits:
        r2_scores_r, _, oof_r, geom_r, fold_labels_r, slide_ids_r = run_rf_cv(
            slide_holdout_folds(rabbit, predictor_names, target_names), predictor_names, target_names,
            plot_per_fold=False)
        plot_r2(r2_scores_r, target_names, cv_label=f'leave-one-slide-out, Rabbit {rabbit.rabbit_id}')
        central_fold_i = len(fold_labels_r) // 2
        plot_fold_comparison_by_slide(central_fold_i, oof_r, geom_r, slide_ids_r, target_names,
                                      fold_labels_r[central_fold_i], value_range=target_value_range)

