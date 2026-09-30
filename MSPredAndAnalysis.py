# Multi-stage MR -> H&E prediction: evaluation plan
#
# Predicts the analysis plan's shared H&E-derived targets (tissue class,
# stain intensity / cell density, tissue proportions) from progressively
# richer MR feature-extraction strategies (Stage 0 baseline voxel intensities
# through Stage 4 from-scratch CNNs), reusing MultiRabAnalysis.py's
# rabbit/slide-holdout CV harness, build_model, and plotting helpers (import
# them rather than duplicating, they're already generic over
# predictor_names/target_names).
#
# Cross-stage comparability: stages operate at different native spatial
# resolutions -- Stage 0/1 predict per MR voxel (0.5mm), Stage 2's RadioDino
# patch embeddings predict per 16px patch (8mm, 224px tile / 14x14 patch
# grid). Scoring each stage against a target at its own native resolution is
# NOT a fair cross-stage comparison: aggregating a target over a coarser
# window (e.g. mean over a patch's voxels) shrinks the target's own variance
# (partly by averaging out registration-error noise), which mechanically
# inflates R² regardless of whether a stage's features actually carry more
# signal. Fix: for the cross-stage comparison numbers, mean-pool every
# stage's per-voxel predictions up to the coarsest shared grid (currently the
# patch grid) and score all stages against the same patch-level H&E target.
# Each stage's native-resolution metric can still be reported separately as
# a diagnostic, just not used for cross-stage comparison.
#
# Patch coverage filtering happens AFTER embedding extraction, not before:
# the ViT's self-attention mixes information across the whole 224x224 tile
# to produce each patch's embedding, so a patch can't be dropped
# pre-extraction without corrupting its neighbors' embeddings too -- the
# model needs the full (possibly zero-padded, since tile_size can exceed the
# real FOV) tile as context regardless of which individual patches end up
# usable. So: flatten ExtractMRFeatures.py's per-tile (14,14,768) embedding
# grid to (196,768) rows, then drop rows whose corresponding H&E/MR
# sub-region lacks real coverage -- reusing the same two coverage
# conventions ExtractMRFeatures.py already applies at the tile level
# (valid_mask's NaN-only check for H&E, the nonzero-fraction check for MR;
# see filter_tiles_by_he_coverage / filter_tiles_by_mr_coverage), just
# computed per patch's 16x16 sub-region instead of per whole tile.
#
# Each flattened, coverage-filtered patch row also needs real-world/pixel
# coordinates (patch (i,j) within a tile -> pixel (x,y) in the slice) so it
# can be matched against the correct H&E sub-region target -- not yet
# computed anywhere; needs a patch-level analog of build_tiles'
# tile_centers()/compute_grid_shape().

import os
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from MultiRabAnalysis import build_model

RabbitTable = namedtuple('RabbitTable', ['rabbit_id', 'predictor_names', 'mr_features', 'target', 'voxel_id'])


def valid_row_mask(mr_features, target):
    """Finite predictors and target -- no zero-exclusion, unlike
    MultiRabAnalysis.valid_voxel_mask. extract_voxelwise_features already
    enforces 'every raw MR contrast nonzero' at extraction time (its own
    mask_volumes check), so every row already in voxel_features_by_id.csv
    satisfies that; re-checking it here would only wrongly catch Stage 1a
    derived columns' legitimate zero values (flat-region gradient magnitude,
    a locally uniform neighborhood's std)."""
    return np.isfinite(np.hstack([mr_features, target[:, None]])).all(axis=1)


def load_rabbit_table(rabbit_id, block_num, base_dir, csv_name, feature_col):
    """One rabbit's Stage 0/1a voxel table -- no slide dimension (unlike
    MultiRabAnalysis.py's Rabbit/Slide), since ExtractMRFeatures.py already
    writes one row per voxel-ID across the whole rabbit/block. Every column
    except voxel_id and feature_col is treated as an MR predictor generically
    -- this doesn't hardcode contrast names, so it works whether or not
    rename_to_standardized_mr_columns has been applied upstream (it just
    changes what predictor_names ends up containing)."""
    path = os.path.join(base_dir, f'R{rabbit_id}', 'Analysis', f'Block{block_num:02d}', csv_name)
    if not os.path.exists(path):
        print(f"No {csv_name} for rabbit {rabbit_id}, skipping.")
        return None
    df = pd.read_csv(path)
    predictor_names = np.array([c for c in df.columns if c not in ('voxel_id', feature_col)])
    return RabbitTable(
        rabbit_id=rabbit_id,
        predictor_names=predictor_names,
        mr_features=df[predictor_names].to_numpy(dtype=float),
        target=df[feature_col].to_numpy(dtype=float),
        voxel_id=df['voxel_id'].to_numpy(),
    )


def predictor_standardization(tables):
    """{rabbit_id: predictor_names} -> common/varying report, same
    common-column-intersection pattern as MultiRabAnalysis.feature_standardization,
    applied to predictor names instead of hne_names/mr_names."""
    name_sets = {t.rabbit_id: set(t.predictor_names) for t in tables}
    common = set.intersection(*name_sets.values())
    union = set.union(*name_sets.values())
    varying = union - common

    print(f"{len(common)} predictor(s) common to all {len(tables)} rabbit(s)")
    if varying:
        print(f"{len(varying)} predictor(s) vary across rabbits:")
        for rabbit_id, names in name_sets.items():
            missing = union - names
            if missing:
                print(f"  Rabbit {rabbit_id} missing: {sorted(missing)}")
    return common


def restrict_tables_to_common_predictors(tables, common):
    """Restrict every table's predictor columns down to `common`, in one
    fixed sorted order shared across all rabbits -- required before vstack-ing
    tables together in rabbit_holdout_folds, since raw column order isn't
    guaranteed to match across rabbits even after standardization (confirmed:
    R23-055's header orders T1w CE Day 3 before T2w Day 3, R24-240's is the
    reverse)."""
    order = np.array(sorted(common))
    restricted = []
    for t in tables:
        idx = [np.where(t.predictor_names == name)[0][0] for name in order]
        restricted.append(t._replace(predictor_names=order, mr_features=t.mr_features[:, idx]))
    return restricted


def classify_predictor_stage(name):
    """'stage1a' for a derived-feature column ('_grad_'/'_nbhd_' --
    ExtractMRFeatures.py's gradient_extra_volumes/neighborhood_extra_volumes
    naming), else 'stage0' (a raw MR contrast column)."""
    return 'stage1a' if ('_grad_' in name or '_nbhd_' in name) else 'stage0'


def select_stage_predictors(tables, stage):
    """Restrict every table's predictors down to one stage's columns.
    'stage0' = raw MR contrasts only, the baseline. 'stage0_1a' = everything
    (Stage 0 raw + Stage 1a derived) -- Stage 1a's gradient/neighborhood-stat
    columns are computed FROM the Stage 0 contrasts, so this is additive
    (progressively richer, matching this file's header comment), not Stage
    1a's derived columns in isolation. Radius/axis-type sub-splits within
    Stage 1a (e.g. 2D-only vs 3D-only, or by radius) are a natural later
    extension of this same function, not built yet."""
    if stage == 'stage0':
        keep = lambda n: classify_predictor_stage(n) == 'stage0'
    elif stage == 'stage0_1a':
        keep = lambda n: True
    else:
        raise ValueError(f"Unknown stage: {stage!r}, expected 'stage0'/'stage0_1a'")

    restricted = []
    for t in tables:
        idx = [i for i, n in enumerate(t.predictor_names) if keep(n)]
        restricted.append(t._replace(predictor_names=t.predictor_names[idx], mr_features=t.mr_features[:, idx]))
    return restricted


def rabbit_holdout_folds(tables):
    """One fold per rabbit: train on every other rabbit's voxels pooled
    together, test on the held-out rabbit's voxels. Simpler than
    MultiRabAnalysis.rabbit_holdout_folds since there's no per-slide pooling
    step -- each table already IS one rabbit's full voxel set."""
    folds = []
    for i, held_out in enumerate(tables):
        train_parts = [t for j, t in enumerate(tables) if j != i]
        train_mr = np.vstack([t.mr_features for t in train_parts])
        train_target = np.concatenate([t.target for t in train_parts])
        folds.append((held_out.rabbit_id, train_mr, train_target, held_out.mr_features, held_out.target))
    return folds


def regression_metrics(y_true, y_pred):
    """R² (can go negative -- worse than predicting the mean; a real
    miscalibration signal Pearson/Spearman would hide), RMSE, MAE, and both
    Pearson r (linear association) and Spearman rho (monotonic association,
    robust to the outlier-prone texture/radiomics ratio features elsewhere in
    this pipeline) -- deliberately not just one correlation metric, since
    they answer different questions about the same prediction."""
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
    }


def run_model_comparison(folds, model_types):
    """Fits each model_type against the same LORO folds. Returns
    {model_type: {'per_fold': [(rabbit_id, metrics_dict), ...], 'pooled': metrics_dict}}
    -- 'pooled' concatenates every fold's held-out (y_true, y_pred) before
    scoring, i.e. one number over every rabbit's out-of-fold voxels at once,
    matching the sketch's single 'Overall' marker per model."""
    results = {}
    for model_type in model_types:
        per_fold = []
        all_true, all_pred = [], []
        for rabbit_id, train_mr, train_target, test_mr, test_target in folds:
            train_mask = valid_row_mask(train_mr, train_target)
            test_mask = valid_row_mask(test_mr, test_target)
            X_train, y_train = train_mr[train_mask], train_target[train_mask]
            X_test, y_test = test_mr[test_mask], test_target[test_mask]

            model = build_model(model_type)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = regression_metrics(y_test, y_pred)
            per_fold.append((rabbit_id, metrics))
            all_true.append(y_test)
            all_pred.append(y_pred)
            print(f"[{model_type}] held out Rabbit {rabbit_id}: n={len(y_test)}  "
                  f"R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                  f"r={metrics['pearson_r']:.3f}  ρ={metrics['spearman_rho']:.3f}")

        pooled = regression_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        print(f"[{model_type}] pooled (all rabbits' held-out voxels): "
              f"R²={pooled['r2']:.3f}  RMSE={pooled['rmse']:.4f}  MAE={pooled['mae']:.4f}  "
              f"r={pooled['pearson_r']:.3f}  ρ={pooled['spearman_rho']:.3f}")
        results[model_type] = {'per_fold': per_fold, 'pooled': pooled}
    return results


METRIC_LABELS = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'pearson_r': 'Pearson r', 'spearman_rho': 'Spearman ρ'}


def plot_metric_comparison(results, model_types, metric_names, stage_label=None):
    """One panel per metric: x-axis = model type, one colored dot per rabbit
    (that rabbit's held-out-fold value) plus one larger black 'Overall' dot
    (the pooled value) -- per Jesse's sketch."""
    rabbit_ids = sorted({rid for r in results.values() for rid, _ in r['per_fold']})
    colors = plt.cm.tab10(np.linspace(0, 1, len(rabbit_ids)))
    rabbit_color = dict(zip(rabbit_ids, colors))

    ncols = len(metric_names)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False,
                             layout='constrained')
    axes = axes.flatten()
    x = np.arange(len(model_types))

    for ax, metric in zip(axes, metric_names):
        for xi, model_type in enumerate(model_types):
            for rabbit_id, metrics in results[model_type]['per_fold']:
                ax.scatter(xi, metrics[metric], color=rabbit_color[rabbit_id], s=60,
                          edgecolor='black', linewidth=0.5, zorder=3)
            ax.scatter(xi, results[model_type]['pooled'][metric], color='black', s=220, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.set_title(METRIC_LABELS.get(metric, metric))

    for ax in axes[len(metric_names):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rabbit_color[rid],
                          markersize=8, label=f'Rabbit {rid}') for rid in rabbit_ids]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                              markersize=12, label='Overall'))
    fig.legend(handles=handles, loc='outside right upper')
    title = 'MR -> HnE regression: metric comparison across model architectures (LORO)'
    if stage_label is not None:
        title += f'  [{stage_label}]'
    fig.suptitle(title)
    plt.show()


if __name__ == '__main__':
    BASE_DIR = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RABBIT_BLOCKS = {"23-055": 7, "24-103": 6, "24-240": 3, "24-082": 5, "24-101": 11, "24-058": 6}
    FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
    MODEL_TYPES = ['rf', 'xgboost', 'linear']
    METRIC_NAMES = ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_rho']
    # 'stage0': raw MR contrasts only (baseline). 'stage0_1a': + Stage 1a's
    # derived gradient/neighborhood-stat columns. Run in this order so the
    # baseline prints/plots first.
    STAGES = ['stage0', 'stage0_1a']

    tables = []
    for rabbit_id, block_num in RABBIT_BLOCKS.items():
        t = load_rabbit_table(rabbit_id, block_num, BASE_DIR, 'voxel_features_by_id.csv', FEATURE_COL)
        if t is not None:
            tables.append(t)

    print("--- Predictor standardization (Stage 0/1a) ---")
    common = predictor_standardization(tables)
    tables = restrict_tables_to_common_predictors(tables, common)

    for stage in STAGES:
        print(f"\n=== {stage} ===")
        stage_tables = select_stage_predictors(tables, stage)
        print(f"{len(stage_tables[0].predictor_names)} predictor(s) for {stage}: "
              f"{list(stage_tables[0].predictor_names)}")

        folds = rabbit_holdout_folds(stage_tables)
        results = run_model_comparison(folds, MODEL_TYPES)
        plot_metric_comparison(results, MODEL_TYPES, METRIC_NAMES, stage_label=stage)
