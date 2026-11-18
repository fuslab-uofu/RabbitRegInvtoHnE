# MR -> H&E analysis of the Stage 2 patch table (RadioDino embeddings per 16x16 patch, from
# ExtractMRFeatures.py): LORO or LOBO regression of each H&E class fraction from the embeddings
# (Stage 2). Patch-mean MR intensity is only used for the diagnostics, not as a model.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from MultiRabAnalysis import build_model
from MSPredAndAnalysis import (CV_HOLDOUT_UNITS, classify_predictor_stage, cv_tables, holdout_folds,
                               load_rabbit_table, per_block_correlations, plot_correlation_heatmap,
                               patch_centre_footprint, plot_metric_comparison, predictor_standardization,
                               print_correlation_tables, restrict_tables_to_common_predictors, results_to_frame,
                               run_model_comparison, save_block_volumes, save_figure, select_stage_predictors,
                               select_target, table_label, valid_row_mask)
from RabbitPathFinder import (class_fraction_col, load_ready_rabbit_blocks, mr_norm_csv_suffix,
                              prediction_run_dir, rabbit_data_dir)


def embedding_pca_model(model_type, predictor_names, n_components):
    """Unfit regressor that standardizes the embedding columns and reduces them to n_components
    PCs, leaving patch-mean intensity columns as they are. Fit inside each LORO fold, so the
    scaling and the PCs come from the training rabbits only."""
    is_embedding = np.array([classify_predictor_stage(n) == 'stage2' for n in predictor_names])
    reduce_embeddings = ColumnTransformer([
        ('intensity', 'passthrough', np.flatnonzero(~is_embedding)),
        ('embedding', make_pipeline(StandardScaler(), PCA(n_components, random_state=0)),
         np.flatnonzero(is_embedding)),
    ])
    return make_pipeline(reduce_embeddings, build_model(model_type))


def rabbit_id_accuracy(tables, n_components=None, n_splits=5):
    """Cross-validated balanced accuracy of telling which rabbit a row came from, from its MR
    features (chance = 1 / number of rabbits): how much rabbit fingerprint the features carry.
    Folds hold out whole slices, because overlapping patches within a slice share pixels."""
    masks = [valid_row_mask(t.mr_features, t.target) for t in tables]
    features = np.vstack([t.mr_features[m] for t, m in zip(tables, masks)])
    rabbit = np.concatenate([np.full(m.sum(), t.rabbit_id) for t, m in zip(tables, masks)])
    rabbit_slice = np.concatenate([[f'{t.rabbit_id}_{z}' for z in t.vz[m]] for t, m in zip(tables, masks)])

    steps = [StandardScaler()]
    if n_components is not None:
        steps.append(PCA(n_components, random_state=0))
    classifier = make_pipeline(*steps, LogisticRegression(max_iter=1000))
    scores = cross_val_score(classifier, features, rabbit, groups=rabbit_slice,
                             cv=GroupKFold(n_splits), scoring='balanced_accuracy')
    return float(scores.mean())


def summarize_loro(metrics):
    """Mean over held-out rabbits (or blocks) of each metric per (stage, model), plus n_pos =
    number of held-out rabbits (or blocks) whose Spearman rho is > 0."""
    per_rabbit = metrics[metrics['held_out'] != 'pooled']
    grouped = per_rabbit.groupby(['stage', 'model'], sort=False)
    summary = grouped[['spearman_rho', 'pearson_r', 'mae', 'r2']].mean()
    summary['n_pos'] = grouped['spearman_rho'].agg(lambda rho: int((rho > 0).sum()))
    return summary


def patch_target_frame(tables, cv_mode='loro'):
    """One row per patch, in table order: rabbit, block, held_out (the cv_mode fold that holds
    the patch out, see CV_HOLDOUT_UNITS), vx/vy/vz and the H&E target."""
    unit_of = CV_HOLDOUT_UNITS[cv_mode]
    return pd.DataFrame({
        'rabbit': np.concatenate([np.full(len(t.target), t.rabbit_id) for t in tables]),
        'block': np.concatenate([np.full(len(t.target), t.block_num) for t in tables]),
        'held_out': np.concatenate([np.full(len(t.target), unit_of(t)) for t in tables]),
        'vx': np.concatenate([t.vx for t in tables]),
        'vy': np.concatenate([t.vy for t in tables]),
        'vz': np.concatenate([t.vz for t in tables]),
        'target': np.concatenate([t.target for t in tables]),
    })


def held_out_predictions(tables, results, label):
    """Each model's held-out prediction for every patch, in patch_target_frame row order:
    one '{label}_{model}' column per model, NaN where a row got no prediction."""
    columns = {}
    for model_type, r in results.items():
        per_rabbit = []
        for t in tables:
            test_mask, y_pred = r['predictions'][(t.rabbit_id, t.block_num)]
            full = np.full(len(t.target), np.nan)
            full[test_mask] = y_pred
            per_rabbit.append(full)
        columns[f'{label}_{model_type}'] = np.concatenate(per_rabbit)
    return pd.DataFrame(columns)


def save_patch_prediction_volumes(predictions, target_col, base_dir, mr_norm_dir, out_dir, cv_mode, patch, stride):
    """NIfTIs (MSPredAndAnalysis.save_block_volumes) from a predictions frame (patch_target_frame
    plus held_out_predictions columns, as saved to <target_col>_patches.csv): every 'stage2_*'
    prediction column as <CV>pred_<target_col>_<column>, and the patch target as
    truth_<target_col>_stage2. Each patch's value fills its central stride x stride block."""
    footprint = patch_centre_footprint(patch, stride)
    columns = {col: f'{cv_mode.upper()}pred_{target_col}_{col}'
               for col in predictions.columns if col.startswith('stage2_')}
    columns['target'] = f'truth_{target_col}_stage2'
    for rabbit_id, rabbit_rows in predictions.groupby('rabbit', sort=False):
        for col, name in columns.items():
            block_rows = [(block_num, rows['vx'].to_numpy(), rows['vy'].to_numpy(), rows['vz'].to_numpy(),
                           rows[col].to_numpy()) for block_num, rows in rabbit_rows.groupby('block')]
            save_block_volumes(rabbit_id, block_rows, base_dir, mr_norm_dir, out_dir, name, footprint)


def prediction_level_summary(predictions, prediction_cols, cv_mode='loro'):
    """Per held-out fold (patch_target_frame's 'held_out'): its mean target, its training rows'
    mean target (the level its model is trained on: the other folds, or for 'lobo' the same
    rabbit's other blocks), and per prediction column the mean prediction, R², and R² once that
    fold's mean prediction error (the offset) is removed."""
    rows = []
    for unit, held_out in predictions.groupby('held_out', sort=False):
        train = predictions['held_out'] != unit
        if cv_mode == 'lobo':
            train &= predictions['rabbit'] == held_out['rabbit'].iloc[0]
        row = {'held_out': unit, 'target_mean': held_out['target'].mean(),
               'train_target_mean': predictions.loc[train, 'target'].mean()}
        for col in prediction_cols:
            predicted = held_out[col].notna()
            y, y_pred = held_out.loc[predicted, 'target'], held_out.loc[predicted, col]
            row[f'{col}_mean'] = y_pred.mean()
            row[f'{col}_r2'] = r2_score(y, y_pred)
            row[f'{col}_r2_offset_removed'] = r2_score(y, y_pred - (y_pred.mean() - y.mean()))
        rows.append(row)
    return pd.DataFrame(rows).set_index('held_out')


def single_patch_grid(t, patch):
    """t's rows from one stride-shifted patch grid only, i.e. without staggering. The grids are
    told apart by (vx, vy) mod patch; the one with the most patches is kept (ties: lowest phase)."""
    phases = np.stack([t.vx % patch, t.vy % patch], axis=1)
    phase_values, counts = np.unique(phases, axis=0, return_counts=True)
    keep = (phases == phase_values[np.argmax(counts)]).all(axis=1)
    return t._replace(mr_features=t.mr_features[keep], target=t.target[keep],
                      vx=t.vx[keep], vy=t.vy[keep], vz=t.vz[keep],
                      targets={col: values[keep] for col, values in t.targets.items()})


def intensity_medians(tables):
    """Median of every predictor within each block (rows = blocks), to show whether one
    block's MR intensities sit at a different level from the others'."""
    return pd.DataFrame({table_label(t): np.nanmedian(t.mr_features, axis=0) for t in tables},
                        index=tables[0].predictor_names).T


if __name__ == '__main__':
    BASE_DIR = rabbit_data_dir()  # this machine's ceph mount (Mac or mara)
    RABBIT_BLOCKS = load_ready_rabbit_blocks()
    # Targets: each class's mean fraction over the patch (one CV run per class). Must match
    # ExtractMRFeatures.py's LABEL_SOURCE, MR_NORM_DIR and PATCH_STRIDE.
    CLASS_LABEL_SOURCE = 'R3_softmax'
    CLASS_NAMES = ['muscle', 'immune_cells', 'necrotic_tissue']
    MR_NORM_DIR = 'Normalized_method2'
    PATCH_STRIDE = 4
    PATCH_SIZE = 16  # RadioDino's patch size (ExtractMRFeatures.py's PATCH_SIZE)
    # True = every stride-shifted patch grid (overlapping patches); False = one non-overlapping grid
    # per block (its most-populated one), to see what staggering adds. Outputs go to stage2_unstaggered_<cv>/.
    STAGGERED = True
    PATCH_CSV_NAME = (f'patch_embeddings_by_id_stride{PATCH_STRIDE}'
                      f'{mr_norm_csv_suffix(MR_NORM_DIR)}_{CLASS_LABEL_SOURCE}.csv')
    # A patch needs at least this fraction of its pixels covered by H&E, and by every MR contrast.
    MIN_HE_COVERAGE = 0.5
    MIN_MR_COVERAGE = 0.5
    MODEL_TYPES = ['rf', 'xgboost', 'linear']
    METRIC_NAMES = ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_rho']
    # Embedding PCs kept per fold; every value is run and reported.
    PCA_COMPONENTS = [10, 25, 50]
    CORR_VALUE = 'spearman_rho'  # or 'pearson_r'
    # 'loro' = hold out a whole rabbit (all its blocks) per fold, train on the other rabbits;
    # 'lobo' = hold out one block, train only on that rabbit's other blocks (multi-block rabbits only).
    CV_MODE = 'loro'
    RUN_RABBIT_ID = True
    RUN_MODELS = True
    SHOW_CORR_PLOTS = False  # heatmap of patch-mean intensity vs each class fraction
    # Outputs go to RadPathPredictions/<CLASS_LABEL_SOURCE>_<MR_NORM_DIR>/stage2[_unstaggered]_<CV_MODE>/ on ceph,
    # next to MSPredAndAnalysis.py's stage folders for the same labels and MR normalization.
    # Rerunning with the same CV_MODE replaces them.

    target_cols = [class_fraction_col(c) for c in CLASS_NAMES]
    tables = []
    for rabbit_id, block_num in RABBIT_BLOCKS:
        t = load_rabbit_table(rabbit_id, block_num, BASE_DIR, PATCH_CSV_NAME, target_cols,
                              min_he_coverage=MIN_HE_COVERAGE, min_mr_coverage=MIN_MR_COVERAGE)
        if t is not None:
            tables.append(t)
    if not STAGGERED:
        n_staggered = {table_label(t): len(t.target) for t in tables}
        tables = [single_patch_grid(t, PATCH_SIZE) for t in tables]
        for t in tables:
            print(f"{table_label(t)}: one patch grid keeps {len(t.target):,}/{n_staggered[table_label(t)]:,} patches")
    # Stage labels and output folder name the grid choice, so both can sit side by side.
    stage_prefix = 'stage2' if STAGGERED else 'stage2_unstaggered'

    print("--- Predictor standardization (Stage 2) ---")
    common = predictor_standardization(tables)
    tables = restrict_tables_to_common_predictors(tables, common)

    print("\n--- Median patch-mean MR intensity per block ---")
    print(intensity_medians(select_stage_predictors(tables, 'stage0')).round(3))

    if RUN_RABBIT_ID:
        # The features alone, so this doesn't depend on which class is the target.
        print(f"\n--- Rabbit-ID balanced accuracy (chance = {1 / len({t.rabbit_id for t in tables}):.2f}) ---")
        print(f"patch-mean intensity: {rabbit_id_accuracy(select_stage_predictors(tables, 'stage0')):.2f}")
        for n_components in PCA_COMPONENTS:
            accuracy = rabbit_id_accuracy(select_stage_predictors(tables, 'stage2'), n_components=n_components)
            print(f"embeddings, top {n_components} PCs: {accuracy:.2f}")

    # One CV metrics CSV for Stage 2 (all PCA sizes and targets, see its 'stage' and 'target'
    # columns), as in MSPredAndAnalysis.py's stage folders, plus one held-out patch predictions
    # CSV per target and one metric figure per target and PCA size.
    stage_dir = Path(prediction_run_dir(CLASS_LABEL_SOURCE, MR_NORM_DIR)) / f'{stage_prefix}_{CV_MODE}'
    pred_dir, fig_dir = stage_dir / 'predictions', stage_dir / 'figures'
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    metrics_path = str(stage_dir / 'metrics.csv')
    all_frames = []  # metrics frames of every target and PCA size run so far
    for target_col in target_cols:
        target_tables = select_target(tables, target_col)

        print(f"\n--- Per-block correlations of patch-mean MR intensity with {target_col} ---")
        corr = per_block_correlations(select_stage_predictors(target_tables, 'stage0'))
        print_correlation_tables(corr, value=CORR_VALUE)
        if SHOW_CORR_PLOTS:
            plot_correlation_heatmap(corr, value=CORR_VALUE, target_label=target_col)
            plt.show()

        if not RUN_MODELS:
            continue
        predictions_path = str(pred_dir / f'{target_col}_patches.csv')
        frames = []  # this target's metrics frames, for its summary
        # Every stage shares the same patches, so predictions are added as columns.
        predictions = patch_target_frame(cv_tables(target_tables, CV_MODE), CV_MODE)

        def run_stage(label, stage_tables, model_builder):
            stage_tables = cv_tables(stage_tables, CV_MODE)
            print(f"\n=== {target_col}: {label} ({len(stage_tables[0].predictor_names)} predictor(s)) ===")
            results = run_model_comparison(holdout_folds(stage_tables, CV_MODE), MODEL_TYPES,
                                           model_builder=model_builder)
            frame = results_to_frame(results, label).assign(target=target_col)
            frames.append(frame)
            all_frames.append(frame)
            pd.concat(all_frames, ignore_index=True).to_csv(metrics_path, index=False)
            for col, values in held_out_predictions(stage_tables, results, label).items():
                predictions[col] = values
            predictions.to_csv(predictions_path, index=False, float_format='%.5g')
            print(f"Saved {CV_MODE.upper()} metrics to {metrics_path} and predictions to {predictions_path}")
            save_figure(plot_metric_comparison(results, MODEL_TYPES, METRIC_NAMES,
                                               stage_label=f'{target_col}, {label}', cv_mode=CV_MODE),
                        str(fig_dir), f'{target_col}_{label}_metrics')
            plt.close('all')

        # Patch-mean raw MR only, on the same patches: the averaging baseline the embeddings must beat.
        run_stage(f'{stage_prefix}_patchmean', select_stage_predictors(target_tables, 'stage0'), build_model)
        stage_tables = select_stage_predictors(target_tables, 'stage2')
        names = stage_tables[0].predictor_names
        for n_components in PCA_COMPONENTS:
            run_stage(f'{stage_prefix}_pca{n_components}', stage_tables,
                      lambda model_type: embedding_pca_model(model_type, names, n_components))

        print(f"\n=== {target_col}: {CV_MODE.upper()} summary, mean over held-out folds ===")
        with pd.option_context('display.width', 200):
            print(summarize_loro(pd.concat(frames, ignore_index=True)).round(3))
            print("\n=== Prediction level per held-out fold, embedding models ===")
            print(prediction_level_summary(
                predictions, [c for c in predictions.columns if c.startswith('stage2_')], CV_MODE).round(3).T)
