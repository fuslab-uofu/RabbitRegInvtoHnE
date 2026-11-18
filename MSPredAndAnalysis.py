# MR -> H&E analysis: per-block feature/target correlations, plus leave-one-rabbit-out (LORO)
# or leave-one-block-out (LOBO) regression over Stage 0 (raw MR) and Stage 0+1a (raw + derived)
# predictors.

import os
import re
from collections import namedtuple
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from MultiRabAnalysis import build_model
from CombineBlockNiftis import mean_over_blocks
from PullMRFeaturesByVoxel import mr_grid_reference
from RabbitPathFinder import (block_analysis_dir, block_label, class_fraction_col, day3end_mr_dir,
                              load_ready_rabbit_blocks, mr_norm_csv_suffix, prediction_run_dir,
                              rabbit_data_dir)

# One analysed rabbit block's rows; a rabbit with several analysed blocks has one table per block.
# targets = {column: values} for every target column loaded; target = the one in use (select_target).
RabbitTable = namedtuple('RabbitTable', ['rabbit_id', 'block_num', 'predictor_names', 'mr_features', 'target',
                                         'vx', 'vy', 'vz', 'targets'], defaults=[None])


def table_label(t):
    """'R23-055 Block05': the block a table's rows come from."""
    return block_label(t.rabbit_id, t.block_num)


def valid_row_mask(mr_features, target):
    """Rows with finite predictors and target. No zero check: extraction already dropped
    zero-MR voxels, and derived features (gradient, std) can legitimately be 0."""
    return np.isfinite(np.hstack([mr_features, target[:, None]])).all(axis=1)


def load_rabbit_table(rabbit_id, block_num, base_dir, csv_name, target_cols,
                      min_he_coverage=None, min_mr_coverage=None):
    """Load one rabbit block's table (one row per voxel, tile or patch, from ExtractMRFeatures.py).
    Every column except vx/vy/vz, the target_cols, the '*_coverage_frac' columns and every H&E
    class '*_fraction' column is an MR predictor (so loading a subset of the classes never makes
    the other classes' fractions predictors). target starts as target_cols[0]; select_target
    switches it. Tile/patch rows below min_he_coverage, or below min_mr_coverage in any contrast,
    are dropped."""
    name = block_label(rabbit_id, block_num)
    path = os.path.join(block_analysis_dir(base_dir, rabbit_id, block_num), csv_name)
    if not os.path.exists(path):
        print(f"No {csv_name} for {name}, skipping.")
        return None
    df = pd.read_csv(path)
    coverage_cols = [c for c in df.columns if c.endswith('_coverage_frac')]
    n_rows = len(df)
    if min_he_coverage is not None:
        df = df[df['he_coverage_frac'] >= min_he_coverage]
    if min_mr_coverage is not None:
        mr_coverage_cols = [c for c in coverage_cols if c != 'he_coverage_frac']
        df = df[(df[mr_coverage_cols] >= min_mr_coverage).all(axis=1)]
    if len(df) < n_rows:
        print(f"{name}: {len(df):,}/{n_rows:,} row(s) kept at coverage thresholds "
              f"H&E {min_he_coverage}, MR {min_mr_coverage}")
    predictor_names = np.array([c for c in df.columns
                                if c not in ('vx', 'vy', 'vz', *target_cols, *coverage_cols)
                                and not c.endswith('_fraction')])
    targets = {col: df[col].to_numpy(dtype=float) for col in target_cols}
    return RabbitTable(
        rabbit_id=rabbit_id,
        block_num=block_num,
        predictor_names=predictor_names,
        mr_features=df[predictor_names].to_numpy(dtype=float),
        target=targets[target_cols[0]],
        vx=df['vx'].to_numpy(),
        vy=df['vy'].to_numpy(),
        vz=df['vz'].to_numpy(),
        targets=targets,
    )


def select_target(tables, target_col):
    """tables with target_col (one of the columns load_rabbit_table loaded) as the target."""
    return [t._replace(target=t.targets[target_col]) for t in tables]


def predictor_standardization(tables):
    """Return the predictor names shared by every block; print any that vary."""
    name_sets = {table_label(t): set(t.predictor_names) for t in tables}
    common = set.intersection(*name_sets.values())
    union = set.union(*name_sets.values())
    varying = union - common

    print(f"{len(common)} predictor(s) common to all {len(tables)} block(s)")
    if varying:
        print(f"{len(varying)} predictor(s) vary across blocks:")
        for label, names in name_sets.items():
            missing = union - names
            if missing:
                print(f"  {label} missing: {sorted(missing)}")
    return common


def restrict_tables_to_common_predictors(tables, common):
    """Keep only `common` predictors, in one sorted order for all blocks (CSV column
    order differs between rabbits, so this is required before stacking them)."""
    order = np.array(sorted(common))
    restricted = []
    for t in tables:
        idx = [np.where(t.predictor_names == name)[0][0] for name in order]
        restricted.append(t._replace(predictor_names=order, mr_features=t.mr_features[:, idx]))
    return restricted


def classify_predictor_stage(name):
    """'stage1a' for derived columns ('_grad_'/'_nbhd_' in the name), 'stage1b' for tile
    radiomics ('_rad_'), 'stage2' for embedding dimensions ('_emb'), else 'stage0' (raw MR
    intensity, or a tile/patch's mean of it)."""
    if '_grad_' in name or '_nbhd_' in name:
        return 'stage1a'
    if '_rad_' in name:
        return 'stage1b'
    if '_emb' in name:
        return 'stage2'
    return 'stage0'


# Predictor classes each stage keeps; stage1a and stage1b include their table's raw/mean MR.
STAGE_SELECTIONS = {
    'stage0': {'stage0'},
    'stage1a': {'stage0', 'stage1a'},
    'stage1b': {'stage0', 'stage1b'},
    'stage1b_mean': {'stage0'},  # tile-mean raw MR only: the averaging baseline for stage1b's radiomics
    'stage2': {'stage2'},
}
# The ExtractMRFeatures.py table each stage reads: stage0/stage1a the voxel table, stage1b a tile
# table (its tile-mean MR columns count as 'stage0' there, so stage1b = tile mean + radiomics),
# stage2 the patch table (analysed by PatchEmbeddingPredAndAnalysis.py).
STAGE_TABLES = {'stage0': 'voxel', 'stage1a': 'voxel', 'stage1b': 'tile', 'stage1b_mean': 'tile', 'stage2': 'patch'}
# Single-neighbourhood Stage 1a stages, 'stage1a_<neighbourhood>' (e.g. 'stage1a_2d_r3'): only that
# neighbourhood's mean and std of each contrast, no raw intensity, to compare spatial scales.
STAGE1A_NEIGHBOURHOODS = ['2d_r1', '2d_r3', '2d_r5', '2d_r7', '3d_r1', '3d_r3']
STAGE_TABLES.update({f'stage1a_{nbhd}': 'voxel' for nbhd in STAGE1A_NEIGHBOURHOODS})


def select_stage_predictors(tables, stage):
    """Keep one stage's predictors, e.g. 'stage0' = raw MR intensity only, 'stage1a' = raw +
    derived, 'stage1a_2d_r3' = 2D r3 neighbourhood mean + std only, 'stage1b' = tile mean +
    radiomics, 'stage2' = embeddings."""
    if stage not in STAGE_TABLES:
        raise ValueError(f"Unknown stage: {stage!r}, expected one of {sorted(STAGE_TABLES)}")
    if stage in STAGE_SELECTIONS:
        def keep(name):
            return classify_predictor_stage(name) in STAGE_SELECTIONS[stage]
    else:
        nbhd = stage.removeprefix('stage1a_')
        def keep(name):
            return name.endswith((f'_nbhd_mean_{nbhd}', f'_nbhd_std_{nbhd}'))

    restricted = []
    for t in tables:
        idx = [i for i, n in enumerate(t.predictor_names) if keep(n)]
        if not idx:
            raise ValueError(f"{table_label(t)} has no predictors for {stage!r}")
        restricted.append(t._replace(predictor_names=t.predictor_names[idx], mr_features=t.mr_features[:, idx]))
    return restricted


def split_predictor_name(name):
    """'T1w CE Day 3_nbhd_mean_3d_r3' -> ('T1w CE Day 3', 'nbhd_mean_3d_r3');
    'T1w CE Day 3_rad_glcm_Contrast' -> ('T1w CE Day 3', 'rad_glcm_Contrast');
    a raw contrast column -> (name, 'raw')."""
    for marker in ('_grad_', '_nbhd_', '_rad_'):
        if marker in name:
            i = name.index(marker)
            return name[:i], name[i + 1:]
    return name, 'raw'


VARIANT_FAMILY_ORDER = ['raw', 'nbhd_mean', 'nbhd_std', 'grad', 'rad']


def variant_sort_key(variant):
    """Table row order: raw, nbhd_mean, nbhd_std (2d before 3d, increasing radius), grad."""
    family = next(f for f in VARIANT_FAMILY_ORDER if variant.startswith(f))
    m = re.search(r'_(2d|3d)_r(\d+)$', variant)
    return (VARIANT_FAMILY_ORDER.index(family),
            m.group(1) if m else '', int(m.group(2)) if m else 0, variant)


def per_block_correlations(tables):
    """One row per (block, predictor): Pearson r and Spearman rho vs the H&E target,
    computed within each block (no model, no pooling across blocks). 'unit' names the block."""
    rows = []
    for t in tables:
        for k, name in enumerate(t.predictor_names):
            x = t.mr_features[:, k]
            ok = valid_row_mask(x[:, None], t.target)
            contrast, variant = split_predictor_name(name)
            rows.append({
                'rabbit': t.rabbit_id, 'block': t.block_num, 'unit': table_label(t),
                'predictor': name, 'contrast': contrast, 'variant': variant,
                'stage': classify_predictor_stage(name), 'n': int(ok.sum()),
                'pearson_r': pearsonr(x[ok], t.target[ok])[0],
                'spearman_rho': spearmanr(x[ok], t.target[ok])[0],
            })
    return pd.DataFrame(rows)


def print_correlation_tables(corr, value='spearman_rho', top_n=25):
    """Print one table per contrast (variants x blocks, + mean and n_pos = #blocks > 0),
    then the top_n predictors by |mean|. value: 'spearman_rho' or 'pearson_r'."""
    unit_order = list(dict.fromkeys(corr['unit']))
    n_units = len(unit_order)
    label = METRIC_LABELS.get(value, value)

    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        for contrast in sorted(corr['contrast'].unique()):
            sub = corr[corr['contrast'] == contrast]
            table = sub.pivot(index='variant', columns='unit', values=value)[unit_order]
            table = table.loc[sorted(table.index, key=variant_sort_key)]
            table['mean'] = table[unit_order].mean(axis=1)
            table['n_pos'] = (table[unit_order] > 0).sum(axis=1)
            print(f"\n=== {label} vs H&E target: {contrast} (n_pos = #blocks > 0, of {n_units}) ===")
            print(table.round(3))

        wide = corr.pivot(index='predictor', columns='unit', values=value)[unit_order]
        summary = pd.DataFrame({
            'stage': [classify_predictor_stage(n) for n in wide.index],
            'mean': wide.mean(axis=1),
            'min': wide.min(axis=1),
            'max': wide.max(axis=1),
            'n_pos': (wide > 0).sum(axis=1),
        }, index=wide.index)
        summary = summary.reindex(summary['mean'].abs().sort_values(ascending=False).index)
        print(f"\n=== Top {top_n} predictors by |mean {label}| across blocks ===")
        print(summary.round(3).head(top_n))


def pooled_feature_pca(tables, n_components=10):
    """Pooled PCA of the standardized MR features over every block's valid voxels. Returns
    (scores: one row per voxel with rabbit, block, vx/vy/vz, target and PC columns; loadings: one row
    per predictor; explained: variance fraction per PC)."""
    masks = [valid_row_mask(t.mr_features, t.target) for t in tables]
    features = np.vstack([t.mr_features[m] for t, m in zip(tables, masks)])
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(StandardScaler().fit_transform(features))
    components = pca.components_.copy()
    # PCA signs are arbitrary: orient PC1 so that it increases with T1w CE intensity.
    if components[0, list(tables[0].predictor_names).index('T1w CE Day 3')] < 0:
        pc_scores[:, 0] *= -1
        components[0] *= -1

    pc_names = [f'PC{k + 1}' for k in range(n_components)]
    scores = pd.DataFrame(pc_scores, columns=pc_names)
    scores.insert(0, 'target', np.concatenate([t.target[m] for t, m in zip(tables, masks)]))
    for axis in ('vz', 'vy', 'vx'):
        scores.insert(0, axis, np.concatenate([getattr(t, axis)[m] for t, m in zip(tables, masks)]))
    scores.insert(0, 'block', np.concatenate([np.full(m.sum(), t.block_num) for t, m in zip(tables, masks)]))
    scores.insert(0, 'rabbit', np.concatenate([np.full(m.sum(), t.rabbit_id) for t, m in zip(tables, masks)]))
    loadings = pd.DataFrame(components.T, index=tables[0].predictor_names, columns=pc_names)
    explained = pd.Series(pca.explained_variance_ratio_, index=pc_names, name='explained_variance_ratio')
    return scores, loadings, explained


# What one cross-validation fold holds out, named per table:
#   'loro' = a whole rabbit (all of its blocks together), trained on every other rabbit;
#   'lobo' = one block, trained ONLY on the same rabbit's other blocks (no other rabbits), i.e.
#            within-animal generalization; only rabbits with several blocks get folds.
CV_HOLDOUT_UNITS = {
    'loro': lambda t: f'R{t.rabbit_id}',
    'lobo': table_label,
}


def cv_tables(tables, cv_mode):
    """The tables cv_mode makes folds for: every table for 'loro'; for 'lobo', only blocks of
    rabbits with several blocks (a single-block rabbit has nothing of its own to train on)."""
    if cv_mode != 'lobo':
        return tables
    n_blocks = pd.Series([t.rabbit_id for t in tables]).value_counts()
    kept = [t for t in tables if n_blocks[t.rabbit_id] > 1]
    dropped = sorted({f'R{t.rabbit_id}' for t in tables if n_blocks[t.rabbit_id] == 1})
    print(f"LOBO (within-rabbit): {len(kept)} block(s) from {len({t.rabbit_id for t in kept})} "
          f"multi-block rabbit(s); single-block rabbit(s) skipped: {dropped}")
    return kept


def holdout_folds(tables, cv_mode='loro'):
    """One fold per held-out unit of cv_mode (see CV_HOLDOUT_UNITS): train on the other
    tables' rows pooled ('lobo': only the held-out block's own rabbit's other blocks), test on
    the held-out tables' rows stacked. Pass cv_tables(tables, cv_mode). Yields (held-out name,
    held-out tables, train_mr, train_target, test_mr, test_target) one fold at a time, so
    only one fold's training matrix is in memory."""
    if cv_mode not in CV_HOLDOUT_UNITS:
        raise ValueError(f"Unknown cv_mode: {cv_mode!r}, expected one of {sorted(CV_HOLDOUT_UNITS)}")
    unit_of = CV_HOLDOUT_UNITS[cv_mode]
    for unit in dict.fromkeys(unit_of(t) for t in tables):
        held_out = [t for t in tables if unit_of(t) == unit]
        train_parts = [t for t in tables if unit_of(t) != unit]
        if cv_mode == 'lobo':
            train_parts = [t for t in train_parts if t.rabbit_id == held_out[0].rabbit_id]
            if not train_parts:
                raise ValueError(f"{unit} has no other block of its rabbit to train on; "
                                 f"pass cv_tables(tables, 'lobo')")
        yield (unit, held_out,
               np.vstack([t.mr_features for t in train_parts]),
               np.concatenate([t.target for t in train_parts]),
               np.vstack([t.mr_features for t in held_out]),
               np.concatenate([t.target for t in held_out]))


def split_predictions_by_table(tables, test_mask, y_pred):
    """Undo a fold's row stacking: per held-out table, (its rows' test mask, its predictions)."""
    masks = np.split(test_mask, np.cumsum([len(t.target) for t in tables])[:-1])
    preds = np.split(y_pred, np.cumsum([m.sum() for m in masks])[:-1])
    return list(zip(masks, preds))


def regression_metrics(y_true, y_pred):
    """R², RMSE, MAE, Pearson r, Spearman rho. R² < 0 means worse than predicting the
    mean (miscalibration that r/rho alone would hide)."""
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
    }


def run_model_comparison(folds, model_types, model_builder=build_model):
    """Fit each model type on the same holdout_folds -> {model: {'per_fold': [(held-out name,
    metrics)], 'pooled': metrics, 'predictions': {(rabbit_id, block_num): (test row mask,
    held-out prediction)}}}; 'pooled' scores all held-out voxels together.
    model_builder(model_type) -> unfit regressor."""
    results = {model_type: {'per_fold': [], 'predictions': {}} for model_type in model_types}
    all_true = {model_type: [] for model_type in model_types}
    all_pred = {model_type: [] for model_type in model_types}
    for held_out_name, held_out_tables, train_mr, train_target, test_mr, test_target in folds:
        train_mask = valid_row_mask(train_mr, train_target)
        test_mask = valid_row_mask(test_mr, test_target)
        X_train, y_train = train_mr[train_mask], train_target[train_mask]
        X_test, y_test = test_mr[test_mask], test_target[test_mask]

        for model_type in model_types:
            model = model_builder(model_type)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = regression_metrics(y_test, y_pred)
            results[model_type]['per_fold'].append((held_out_name, metrics))
            for t, prediction in zip(held_out_tables,
                                     split_predictions_by_table(held_out_tables, test_mask, y_pred)):
                results[model_type]['predictions'][(t.rabbit_id, t.block_num)] = prediction
            all_true[model_type].append(y_test)
            all_pred[model_type].append(y_pred)
            print(f"[{model_type}] held out {held_out_name}: n={len(y_test)}  "
                  f"R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                  f"r={metrics['pearson_r']:.3f}  ρ={metrics['spearman_rho']:.3f}")

    for model_type in model_types:
        pooled = regression_metrics(np.concatenate(all_true[model_type]),
                                    np.concatenate(all_pred[model_type]))
        print(f"[{model_type}] pooled (every fold's held-out voxels): "
              f"R²={pooled['r2']:.3f}  RMSE={pooled['rmse']:.4f}  MAE={pooled['mae']:.4f}  "
              f"r={pooled['pearson_r']:.3f}  ρ={pooled['spearman_rho']:.3f}")
        results[model_type]['pooled'] = pooled
    return results


def results_to_frame(results, stage):
    """Flatten run_model_comparison output to one row per (model, held-out rabbit or block),
    plus a 'pooled' row per model, with one column per metric."""
    rows = []
    for model_type, r in results.items():
        for held_out_name, metrics in r['per_fold'] + [('pooled', r['pooled'])]:
            rows.append({'stage': stage, 'model': model_type, 'held_out': held_out_name, **metrics})
    return pd.DataFrame(rows)


def paint_footprints(volume, size, corner_offset):
    """Copy of a row-voxel volume (one value at each tile or patch's row voxel, NaN elsewhere) with
    each value painted over a size x size square on axes 0 and 2, one axis-1 slice thick, whose
    corner is corner_offset voxels before the row voxel on both axes; clipped at the volume edge.
    Stops if two squares overlap."""
    filled = np.full(volume.shape, np.nan, dtype=volume.dtype)
    for a0, a1, a2 in np.argwhere(np.isfinite(volume)):
        x0, y0 = a0 - corner_offset, a2 - corner_offset
        footprint = filled[max(x0, 0):min(x0 + size, volume.shape[0]), a1,
                           max(y0, 0):min(y0 + size, volume.shape[2])]
        if np.isfinite(footprint).any():
            raise ValueError(f"Square at voxel {(a0, a1, a2)} overlaps an already painted one")
        footprint[:] = volume[a0, a1, a2]
    return filled


def tile_footprint(tile_size):
    """paint_footprints (size, corner_offset) covering a whole tile: its row voxel sits
    (tile_size - 1) // 2 from the tile's corner (extract_mri_embedding.patch_pixel_center)."""
    return tile_size, (tile_size - 1) // 2


def patch_centre_footprint(patch, stride):
    """paint_footprints (size, corner_offset) covering a patch's central stride x stride block.
    Patches on a stride lattice have these blocks tile the plane, each voxel in exactly one."""
    return stride, (patch - 1) // 2 - (patch - stride) // 2


def save_block_volumes(rabbit_id, block_rows, base_dir, mr_norm_dir, out_dir, name, footprint=None):
    """Write one rabbit's values as a NIfTI per block on its MR grid (the grid the feature tables
    index; NaN where there is no value), for viewing in Slicer over the MR volumes read from
    mr_norm_dir. block_rows = [(block_num, vx, vy, vz, values)]; each value goes at its row voxel,
    or over footprint = (size, corner_offset), see paint_footprints. Files are
    R<id>_BlockNN_<name>.nii.gz, plus their mean R<id>_AllBlocks_<name>.nii.gz for several blocks."""
    target = mr_grid_reference(day3end_mr_dir(base_dir, rabbit_id, mr_norm_dir))
    block_volumes = []
    for block_num, vx, vy, vz, values in block_rows:
        volume = np.full(target.shape, np.nan, dtype=np.float32)
        # CSV coordinates are (axis0, axis2, axis1) indices, see extract_voxelwise_features.
        volume[vx, vz, vy] = values
        if footprint is not None:
            volume = paint_footprints(volume, *footprint)
        out_path = os.path.join(out_dir, f'R{rabbit_id}_Block{block_num:02d}_{name}.nii.gz')
        nib.save(nib.Nifti1Image(volume, target.affine), out_path)
        print(f"Saved {out_path}")
        block_volumes.append(volume)
    if len(block_volumes) > 1:
        out_path = os.path.join(out_dir, f'R{rabbit_id}_AllBlocks_{name}.nii.gz')
        nib.save(nib.Nifti1Image(mean_over_blocks(np.stack(block_volumes)), target.affine), out_path)
        print(f"Saved {out_path}")


def save_prediction_volumes(tables, results, target_col, stage, base_dir, mr_norm_dir, out_dir, cv_mode='loro',
                            footprint=None):
    """save_block_volumes of each model's held-out prediction of target_col for every block, as
    <CV>pred_<target_col>_<stage>_<model>. footprint (tile tables, see tile_footprint): each
    tile's prediction fills its footprint instead of its single row voxel."""
    tables_by_rabbit = {}
    for t in tables:
        tables_by_rabbit.setdefault(t.rabbit_id, []).append(t)
    for rabbit_id, rabbit_tables in tables_by_rabbit.items():
        for model_type, r in results.items():
            block_rows = []
            for t in rabbit_tables:
                test_mask, y_pred = r['predictions'][(t.rabbit_id, t.block_num)]
                block_rows.append((t.block_num, t.vx[test_mask], t.vy[test_mask], t.vz[test_mask], y_pred))
            save_block_volumes(rabbit_id, block_rows, base_dir, mr_norm_dir, out_dir,
                               f'{cv_mode.upper()}pred_{target_col}_{stage}_{model_type}', footprint)


def save_target_volumes(tables, target_col, stage, base_dir, mr_norm_dir, out_dir, footprint=None):
    """save_block_volumes of every block's target (what the models were trained to predict) as
    truth_<target_col>_<stage>, painted over the same footprint as the predictions."""
    tables_by_rabbit = {}
    for t in tables:
        tables_by_rabbit.setdefault(t.rabbit_id, []).append(t)
    for rabbit_id, rabbit_tables in tables_by_rabbit.items():
        block_rows = [(t.block_num, t.vx, t.vy, t.vz, t.target) for t in rabbit_tables]
        save_block_volumes(rabbit_id, block_rows, base_dir, mr_norm_dir, out_dir,
                           f'truth_{target_col}_{stage}', footprint)


def voxel_class_labels(vx, vy, vz, label_path, target_grid_path):
    """Class label at each table row's voxel, from a label NIfTI (e.g.
    ProjectClassPredictionsToMR.py's output). Table coordinates index the rabbit's H&E target
    (target_grid_path, its *_RL_iter1.nii.gz); each voxel is found in the label map through
    world coordinates, since the two can be the same grid stored in a different axis order
    (e.g. canonical RAS labels vs the raw-orientation target). Stops if a voxel doesn't land
    exactly on a label voxel, i.e. the grids really differ."""
    labels = nib.load(str(label_path))
    target = nib.load(str(target_grid_path))
    # CSV coordinates are (axis0, axis2, axis1) indices, see extract_voxelwise_features.
    target_idx = np.stack([vx, vz, vy, np.ones(len(vx))]).astype(float)
    label_idx = (np.linalg.inv(labels.affine) @ target.affine @ target_idx)[:3]
    rounded = np.round(label_idx).astype(np.int64)
    on_grid = np.allclose(label_idx, rounded, atol=1e-3)
    in_bounds = all(((rounded[a] >= 0) & (rounded[a] < labels.shape[a])).all() for a in range(3))
    if not (on_grid and in_bounds):
        raise ValueError(f"Label map grid {labels.shape} isn't the H&E target grid {target.shape} "
                         f"(voxels don't line up):\n  {label_path}\n  {target_grid_path}")
    return np.asanyarray(labels.dataobj)[rounded[0], rounded[1], rounded[2]]


def class_fraction_tables(tables, class_name, label_source, base_dir):
    """tables with each row's target replaced by its voxel's fraction of class_name, read from
    ProjectClassPredictionsToMR.py's MR-grid map for label_source (NaN where no H&E covers the
    voxel; those rows are dropped by valid_row_mask)."""
    out = []
    for t in tables:
        prefix = f'R{t.rabbit_id}_Block{t.block_num:02d}'
        fraction_path = (Path(__file__).parent / 'TestData' / f'{prefix}_ClassPredictions_{label_source}'
                         / f'{prefix}_MR_{class_name}_fraction.nii.gz')
        target_grid_path = os.path.join(block_analysis_dir(base_dir, t.rabbit_id, t.block_num),
                                        f'{prefix}_RL_iter1.nii.gz')
        fraction = voxel_class_labels(t.vx, t.vy, t.vz, fraction_path, target_grid_path)
        out.append(t._replace(target=np.asarray(fraction, dtype=float)))
    return out


METRIC_LABELS = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'pearson_r': 'Pearson r', 'spearman_rho': 'Spearman ρ'}

# A badly offset held-out fold's R² can reach the -100s and flatten every other fold onto 0, so
# the R² axis stops here and lower values sit on it as down-triangles labelled with their value.
R2_AXIS_FLOOR = -1.0


def unit_display_names(corr):
    """{corr 'unit': figure label}: the rabbit ID ('R23-055'), plus ' B<block>' only when that
    rabbit has several blocks in corr."""
    units = corr.drop_duplicates('unit')
    several_blocks = units['rabbit'].duplicated(keep=False)
    return {unit: f'R{rid}' + (f' B{block}' if several else '')
            for unit, rid, block, several in zip(units['unit'], units['rabbit'], units['block'],
                                                 several_blocks)}


def unit_colors(names):
    """One color per rabbit or block name, assigned over the sorted names so every figure
    agrees; tab20 once there are more names than tab10 has colors."""
    names = sorted(names)
    cmap = plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20
    return dict(zip(names, cmap(np.linspace(0, 1, len(names)))))


def unit_legend_handles(unit_color, summary_label=None):
    """Legend entries: one dot per rabbit or block, plus a large black dot named summary_label if given."""
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, label=name) for name, color in unit_color.items()]
    if summary_label is not None:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                  markersize=12, label=summary_label))
    return handles


def scatter_above_floor(ax, x, value, floor, label_row=0, **scatter_kwargs):
    """ax.scatter of one value; a value below floor is drawn on the floor as a down-triangle
    labelled with its true value, the label raised by label_row lines so labels at one x don't
    overlap. Returns whether the value was below floor."""
    if not value < floor:  # also plots NaN as usual (nothing drawn)
        ax.scatter(x, value, **scatter_kwargs)
        return False
    ax.scatter(x, floor, marker='v', **scatter_kwargs)
    ax.annotate(f'{value:.3g}', (x, floor), xytext=(9, 9 * label_row), textcoords='offset points',
                fontsize=7, va='center', color=scatter_kwargs.get('color', 'black'))
    return True


def plot_metric_comparison(results, model_types, metric_names, stage_label=None, cv_mode='loro'):
    """One panel per metric: x = model type, a colored dot per held-out rabbit or block,
    plus a large black dot for the pooled 'Overall' value. R² is cut off at R2_AXIS_FLOOR."""
    unit_color = unit_colors({name for r in results.values() for name, _ in r['per_fold']})

    ncols = len(metric_names)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False,
                             layout='constrained')
    axes = axes.flatten()
    x = np.arange(len(model_types))

    for ax, metric in zip(axes, metric_names):
        floor = R2_AXIS_FLOOR if metric == 'r2' else -np.inf
        for xi, model_type in enumerate(model_types):
            n_clipped = 0
            for held_out_name, metrics in results[model_type]['per_fold']:
                n_clipped += scatter_above_floor(ax, xi, metrics[metric], floor, label_row=n_clipped,
                                                 color=unit_color[held_out_name], s=60,
                                                 edgecolor='black', linewidth=0.5, zorder=3)
            scatter_above_floor(ax, xi, results[model_type]['pooled'][metric], floor, label_row=n_clipped,
                                color='black', s=220, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.set_title(METRIC_LABELS.get(metric, metric))

    for ax in axes[len(metric_names):]:
        ax.set_visible(False)

    fig.legend(handles=unit_legend_handles(unit_color, 'Overall'), loc='outside right upper')
    title = f'MR -> HnE regression: metric comparison across model architectures ({cv_mode.upper()})'
    if stage_label is not None:
        title += f'  [{stage_label}]'
    fig.suptitle(title)
    return fig


def save_figure(fig, fig_dir, name):
    """Write fig to fig_dir/name.png."""
    fig.savefig(os.path.join(fig_dir, f'{name}.png'), dpi=200, bbox_inches='tight')


def plot_feature_correlations(corr, value='spearman_rho', raw_only=False, target_label='H&E target'):
    """Plot of per_block_correlations: one panel per contrast, x = feature variant, a colored
    dot per block plus a large black dot for the across-block mean. raw_only: a single
    panel of the raw contrasts instead. target_label names the target in the title."""
    unit_color = unit_colors(corr['unit'].unique())
    label = METRIC_LABELS.get(value, value)
    contrasts = sorted(corr['contrast'].unique())
    if raw_only:
        panels = [('raw intensity', corr[corr['variant'] == 'raw'], 'contrast', contrasts)]
    else:
        variants = sorted(corr['variant'].unique(), key=variant_sort_key)
        panels = [(c, corr[corr['contrast'] == c], 'variant', variants) for c in contrasts]

    n_x = len(panels[0][3])
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(9, 0.6 * n_x + 3), 3 * len(panels) + 1),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    for ax, (panel_title, sub, x_col, x_order) in zip(axes[:, 0], panels):
        wide = sub.pivot(index=x_col, columns='unit', values=value).reindex(x_order)
        x = np.arange(len(x_order))
        for unit in wide.columns:
            ax.scatter(x, wide[unit], color=unit_color[unit], s=60,
                       edgecolor='black', linewidth=0.5, zorder=3)
        ax.scatter(x, wide.mean(axis=1), color='black', s=220, zorder=2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_order, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.set_title(panel_title)

    fig.legend(handles=unit_legend_handles(unit_color, 'Mean'), loc='outside right upper')
    fig.suptitle(f'{label} of each MR feature with {target_label}, within each block')
    return fig


@plt.rc_context({'font.size': 12})
def plot_correlation_heatmap(corr, value='spearman_rho', variants=('raw',), target_label='H&E target'):
    """Heatmap of per_block_correlations: rows = `variants` within each contrast, columns =
    blocks plus an across-block mean column, each cell colored by and labeled with its value.
    target_label names the target in the title."""
    unit_order = list(dict.fromkeys(corr['unit']))
    label = METRIC_LABELS.get(value, value)
    contrasts = sorted(corr['contrast'].unique())
    rows = [(c, v) for c in contrasts for v in variants]

    wide = corr.pivot(index=['contrast', 'variant'], columns='unit', values=value)
    wide = wide.loc[rows, unit_order]
    grid = np.hstack([wide.to_numpy(), wide.mean(axis=1).to_numpy()[:, None]])

    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 3, 0.35 * n_rows + 1.2), layout='constrained')
    # Full correlation range, so the color strength reads as the actual strength of association.
    im = ax.imshow(grid, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(grid[i, j]) > 0.6 else 'black')

    # White gaps set the mean column apart from the blocks, and each contrast from the next.
    ax.axvline(len(unit_order) - 0.5, color='white', linewidth=4)
    for k in range(1, len(contrasts)):
        ax.axhline(k * len(variants) - 0.5, color='white', linewidth=4)

    ax.set_xticks(np.arange(n_cols))
    display_names = unit_display_names(corr)
    ax.set_xticklabels([display_names[unit] for unit in unit_order] + ['Mean'])
    if len(variants) == 1:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(contrasts)
    else:
        # Contrast names on the left, centered on their group; variant names on the right.
        ax.set_yticks(np.arange(len(contrasts)) * len(variants) + (len(variants) - 1) / 2)
        ax.set_yticklabels(contrasts)
        variant_axis = ax.secondary_yaxis('right')
        variant_axis.set_yticks(np.arange(n_rows))
        variant_axis.set_yticklabels([v for _, v in rows])
        variant_axis.tick_params(length=0)
        variant_axis.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    fig.colorbar(im, ax=ax, label=label)
    if tuple(variants) == ('raw',):
        fig.suptitle(f'{label} of Raw MR Voxel Intensity with {target_label}')
    else:
        fig.suptitle(f'{label} of MR features with {target_label}, within each block')
    return fig


def plot_raw_feature_scatter(tables, corr, pooled=False, points_per_rabbit=3000,
                             target_label='H&E target'):
    """Raw MR intensity (x) vs the target (y), named target_label on the plot. Default: density
    grid, rows = blocks, columns = contrasts, each panel annotated with its within-block rho
    and r. pooled: one panel per contrast, a random subsample of every block's voxels colored
    by block, with a large dot at each block's mean."""
    raw_names = [n for n in tables[0].predictor_names if classify_predictor_stage(n) == 'stage0']
    unit_color = unit_colors([table_label(t) for t in tables])
    stats = corr.set_index(['unit', 'predictor'])

    def valid_xy(t, name):
        x = t.mr_features[:, list(t.predictor_names).index(name)]
        ok = valid_row_mask(x[:, None], t.target)
        return x[ok], t.target[ok]

    if pooled:
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(1, len(raw_names), figsize=(4.5 * len(raw_names), 4.5),
                                 squeeze=False, sharey=True, layout='constrained')
        for ax, name in zip(axes[0], raw_names):
            xs, ys, colors = [], [], []
            for t in tables:
                x, y = valid_xy(t, name)
                keep = rng.choice(len(x), size=min(points_per_rabbit, len(x)), replace=False)
                xs.append(x[keep])
                ys.append(y[keep])
                colors.append(np.tile(unit_color[table_label(t)], (len(keep), 1)))
                ax.scatter(x.mean(), y.mean(), color=unit_color[table_label(t)], s=160,
                           edgecolor='black', linewidth=1.2, zorder=3)
            # Shuffled so no block is drawn on top of all the others.
            order = rng.permutation(sum(len(x) for x in xs))
            ax.scatter(np.concatenate(xs)[order], np.concatenate(ys)[order],
                       c=np.concatenate(colors)[order], s=3, alpha=0.25, linewidths=0, zorder=2)
            ax.set_title(name)
            ax.set_xlabel('MR intensity')
        axes[0, 0].set_ylabel(target_label)
        fig.legend(handles=unit_legend_handles(unit_color), loc='outside right upper')
        fig.suptitle(f'Raw MR intensity vs {target_label}, all blocks (large dot = block mean)')
        return fig

    fig, axes = plt.subplots(len(tables), len(raw_names),
                             figsize=(3 * len(raw_names) + 1, 2.4 * len(tables) + 1),
                             squeeze=False, sharex='col', sharey=True, layout='constrained')
    densities = []
    for row_axes, t in zip(axes, tables):
        for ax, name in zip(row_axes, raw_names):
            x, y = valid_xy(t, name)
            densities.append(ax.hexbin(x, y, gridsize=40, cmap='Blues', mincnt=1))
            s = stats.loc[(table_label(t), name)]
            ax.text(0.03, 0.97, f"ρ={s['spearman_rho']:.2f}  r={s['pearson_r']:.2f}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
        row_axes[0].set_ylabel(f'{table_label(t)}\n{target_label}')
    for ax, name in zip(axes[0], raw_names):
        ax.set_title(name)
    for ax in axes[-1]:
        ax.set_xlabel('MR intensity')
    # One log color scale for every panel, so density is comparable across the grid.
    shared_norm = LogNorm(vmin=1, vmax=max(d.get_array().max() for d in densities))
    for d in densities:
        d.set_norm(shared_norm)
    fig.colorbar(densities[0], ax=axes, location='right', fraction=0.02, pad=0.01,
                 label='voxels per bin')
    fig.suptitle(f'Raw MR intensity vs {target_label}, within each block')
    return fig


if __name__ == '__main__':
    BASE_DIR = rabbit_data_dir()  # this machine's ceph mount (Mac or mara)
    RABBIT_BLOCKS = load_ready_rabbit_blocks()
    FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
    # What to predict: 'he_feature' = FEATURE_COL from older (RL_iter1-target) voxel tables;
    # 'class_fractions' = each class's per-voxel fraction column (one CV run per class).
    TARGET = 'class_fractions'
    CLASS_LABEL_SOURCE = 'R3_softmax'  # ExtractMRFeatures.py's LABEL_SOURCE
    CLASS_NAMES = ['muscle', 'immune_cells', 'necrotic_tissue']
    # Must match ExtractMRFeatures.py's MR_NORM_DIR (CSV names carry it and the label source).
    MR_NORM_DIR = 'Normalized_method2'
    CSV_SUFFIX = mr_norm_csv_suffix(MR_NORM_DIR) + (f'_{CLASS_LABEL_SOURCE}' if TARGET == 'class_fractions' else '')
    TARGET_COLS = ([FEATURE_COL] if TARGET == 'he_feature' else
                   [class_fraction_col(c) for c in CLASS_NAMES])
    MODEL_TYPES = ['rf', 'xgboost', 'linear']
    METRIC_NAMES = ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_rho']
    # Stages to model, in order (Stage 2 runs in PatchEmbeddingPredAndAnalysis.py):
    #   'stage0'  = raw MR per voxel (the baseline);
    #   'stage1a' = raw MR + neighbourhood/gradient features per voxel;
    #   'stage1a_<nbhd>' = one neighbourhood's mean + std per voxel, nbhd in STAGE1A_NEIGHBOURHOODS
    #               (e.g. 'stage1a_2d_r3'; add them all with [f'stage1a_{n}' for n in STAGE1A_NEIGHBOURHOODS]);
    #   'stage1b' = tile-mean MR + radiomics per TILE1B_SIZE tile ('linear' is unreliable here:
    #               plain least squares on hundreds of correlated radiomics; read rf/xgboost);
    #   'stage1b_mean' = tile-mean MR only on the same tiles, the baseline stage1b's radiomics must beat.
    STAGES = ['stage0', 'stage1a', 'stage1b']
    TILE1B_SIZE = 16  # one of ExtractMRFeatures.py's TILE1B_SIZES (3, 7, 11, 16)
    STAGE_CSV_NAMES = {'voxel': f'voxel_features_by_id{CSV_SUFFIX}.csv',
                       'tile': f'tile_features_by_id_size{TILE1B_SIZE}{CSV_SUFFIX}.csv'}
    # Correlation tables (voxel table) always run; CV models only when RUN_MODELS is True.
    CORR_VALUE = 'spearman_rho'  # or 'pearson_r'
    RUN_MODELS = True
    # 'loro' = hold out a whole rabbit (all its blocks) per fold, train on the other rabbits;
    # 'lobo' = hold out one block, train only on that rabbit's other blocks (multi-block rabbits only).
    CV_MODE = 'loro'
    RUN_PCA = True  # pooled PCA of the voxel table's MR features, saved for building figures
    SHOW_CORR_PLOTS = True  # correlation dot plot + raw MR vs H&E scatter (per block and pooled)
    # Outputs go to RadPathPredictions/<label source or FEATURE_COL>_<MR_NORM_DIR>/ on ceph: one
    # <stage>_<CV_MODE> folder per stage (metrics, predictions/, figures/) plus correlations_pca/.
    # Rerunning a stage with the same CV_MODE replaces that folder's outputs.
    SHOW_FIGURES = False  # False = save figures only, so the run never pauses on plot windows

    def load_feature_tables(csv_name):
        """Every ready block's table from csv_name, restricted to the predictors all blocks share."""
        loaded = []
        for rabbit_id, block_num in RABBIT_BLOCKS:
            t = load_rabbit_table(rabbit_id, block_num, BASE_DIR, csv_name, TARGET_COLS)
            if t is not None:
                loaded.append(t)
        print(f"--- Predictor standardization ({csv_name}) ---")
        return restrict_tables_to_common_predictors(loaded, predictor_standardization(loaded))

    if 'stage2' in STAGES:
        raise ValueError("Stage 2 runs in PatchEmbeddingPredAndAnalysis.py; remove it from STAGES here")
    # Only the tables some stage (or the voxel correlations/PCA) needs are loaded.
    needed_tables = {STAGE_TABLES[s] for s in STAGES} | {'voxel'}
    tables_by_kind = {kind: load_feature_tables(STAGE_CSV_NAMES[kind]) for kind in sorted(needed_tables)}
    voxel_tables = tables_by_kind['voxel']

    run_dir = Path(prediction_run_dir(CLASS_LABEL_SOURCE if TARGET == 'class_fractions' else FEATURE_COL,
                                      MR_NORM_DIR))
    # Voxel-table outputs made once per run rather than per stage.
    corr_pca_dir = str(run_dir / 'correlations_pca')
    os.makedirs(corr_pca_dir, exist_ok=True)

    # (target name, target column, also the output file name prefix)
    if TARGET == 'he_feature':
        targets = [(FEATURE_COL, FEATURE_COL)]
    elif TARGET == 'class_fractions':
        targets = [(f'{c} fraction ({CLASS_LABEL_SOURCE})', class_fraction_col(c)) for c in CLASS_NAMES]
    else:
        raise ValueError(f"Unknown TARGET {TARGET!r}")

    for target_name, target_col in targets:
        target_tables = select_target(voxel_tables, target_col)
        print(f"\n--- Per-block correlations with {target_name} (voxel table: Stage 0 raw + Stage 1a derived) ---")
        corr = per_block_correlations(target_tables)
        print_correlation_tables(corr, value=CORR_VALUE)
        if SHOW_CORR_PLOTS:
            save_figure(plot_feature_correlations(corr, value=CORR_VALUE, target_label=target_name),
                        corr_pca_dir, f'{target_col}_corr_dots')
            save_figure(plot_correlation_heatmap(corr, value=CORR_VALUE, target_label=target_name),
                        corr_pca_dir, f'{target_col}_corr_heatmap_raw')
            save_figure(plot_correlation_heatmap(corr, value=CORR_VALUE,
                                                 variants=('raw', 'nbhd_std_2d_r3', 'grad_3d'),
                                                 target_label=target_name),
                        corr_pca_dir, f'{target_col}_corr_heatmap_variants')
            save_figure(plot_raw_feature_scatter(target_tables, corr, target_label=target_name),
                        corr_pca_dir, f'{target_col}_scatter_per_rabbit')
            save_figure(plot_raw_feature_scatter(target_tables, corr, pooled=True,
                                                 target_label=target_name),
                        corr_pca_dir, f'{target_col}_scatter_pooled')
            plt.show() if SHOW_FIGURES else plt.close('all')

    if RUN_PCA:
        scores, loadings, explained = pooled_feature_pca(voxel_tables)
        scores.to_csv(os.path.join(corr_pca_dir, 'pca_scores.csv'), index=False, float_format='%.5g')
        loadings.to_csv(os.path.join(corr_pca_dir, 'pca_loadings.csv'), index_label='predictor')
        explained.to_csv(os.path.join(corr_pca_dir, 'pca_explained_variance.csv'), index_label='component')
        print(f"Saved pooled PCA scores, loadings and explained variance to {corr_pca_dir}")

    if RUN_MODELS:
        # Per stage folder: one CV metrics CSV (all targets, see its 'target' column), held-out
        # prediction NIfTIs and metric figures, for building figures without refitting.
        stage_frames = {}  # stage label -> metrics frames of the targets run so far
        for target_name, target_col in targets:
            for stage in STAGES:
                # Names the tile size for stage1b, so runs at different sizes keep separate outputs.
                stage_label = f'{stage}_tile{TILE1B_SIZE}' if STAGE_TABLES[stage] == 'tile' else stage
                stage_dir = run_dir / f'{stage_label}_{CV_MODE}'
                pred_dir, fig_dir = str(stage_dir / 'predictions'), str(stage_dir / 'figures')
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(fig_dir, exist_ok=True)
                print(f"\n=== {target_name}: {stage_label} ===")
                table = select_target(tables_by_kind[STAGE_TABLES[stage]], target_col)
                stage_tables = cv_tables(select_stage_predictors(table, stage), CV_MODE)
                print(f"{len(stage_tables[0].predictor_names)} predictor(s) for {stage_label}")

                folds = holdout_folds(stage_tables, CV_MODE)
                results = run_model_comparison(folds, MODEL_TYPES)
                frames = stage_frames.setdefault(stage_label, [])
                frames.append(results_to_frame(results, stage_label).assign(target=target_name))
                metrics_path = str(stage_dir / 'metrics.csv')
                pd.concat(frames, ignore_index=True).to_csv(metrics_path, index=False)
                print(f"Saved {CV_MODE.upper()} metrics to {metrics_path}")
                # Tile stages are saved at each tile's row voxel; filled maps are made on demand from these.
                save_prediction_volumes(stage_tables, results, target_col, stage_label, BASE_DIR, MR_NORM_DIR,
                                        pred_dir, CV_MODE)
                save_figure(plot_metric_comparison(results, MODEL_TYPES, METRIC_NAMES,
                                                   stage_label=f'{target_name}, {stage_label}',
                                                   cv_mode=CV_MODE),
                            fig_dir, f'{target_col}_metrics')
                plt.show() if SHOW_FIGURES else plt.close('all')
