# MR -> H&E analysis of the Stage 2 patch table (RadioDino embeddings per 16x16 patch, from
# ExtractMRFeatures.py): LORO or LOBO regression of each H&E class fraction from the embeddings
# vs a patch-mean intensity baseline.

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
from MSPredAndAnalysis import (CV_HOLDOUT_UNITS, classify_predictor_stage, holdout_folds,
                               load_rabbit_table, per_block_correlations, plot_correlation_heatmap,
                               predictor_standardization, print_correlation_tables,
                               restrict_tables_to_common_predictors, results_to_frame,
                               run_model_comparison, select_stage_predictors, select_target,
                               table_label, valid_row_mask)
from RabbitPathFinder import class_fraction_col, load_ready_rabbit_blocks, mr_norm_csv_suffix, rabbit_data_dir


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


def prediction_level_summary(predictions, prediction_cols):
    """Per held-out fold (patch_target_frame's 'held_out'): its mean target, the rest's mean
    target (the level its model is trained on), and per prediction column the mean prediction,
    R², and R² once that fold's mean prediction error (the offset) is removed."""
    rows = []
    for unit, held_out in predictions.groupby('held_out', sort=False):
        row = {'held_out': unit, 'target_mean': held_out['target'].mean(),
               'train_target_mean': predictions.loc[predictions['held_out'] != unit, 'target'].mean()}
        for col in prediction_cols:
            predicted = held_out[col].notna()
            y, y_pred = held_out.loc[predicted, 'target'], held_out.loc[predicted, col]
            row[f'{col}_mean'] = y_pred.mean()
            row[f'{col}_r2'] = r2_score(y, y_pred)
            row[f'{col}_r2_offset_removed'] = r2_score(y, y_pred - (y_pred.mean() - y.mean()))
        rows.append(row)
    return pd.DataFrame(rows).set_index('held_out')


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
    PATCH_CSV_NAME = (f'patch_embeddings_by_id_stride{PATCH_STRIDE}'
                      f'{mr_norm_csv_suffix(MR_NORM_DIR)}_{CLASS_LABEL_SOURCE}.csv')
    # A patch needs at least this fraction of its pixels covered by H&E, and by every MR contrast.
    MIN_HE_COVERAGE = 0.5
    MIN_MR_COVERAGE = 0.5
    MODEL_TYPES = ['rf', 'xgboost', 'linear']
    # Embedding PCs kept per fold; every value is run and reported.
    PCA_COMPONENTS = [10, 25, 50]
    CORR_VALUE = 'spearman_rho'  # or 'pearson_r'
    # 'loro' = hold out a whole rabbit (all its blocks) per fold; 'lobo' = hold out one block.
    CV_MODE = 'loro'
    RUN_RABBIT_ID = True
    RUN_MODELS = True
    SHOW_CORR_PLOTS = False  # heatmap of patch-mean intensity vs each class fraction
    # Outputs go to TestData/PatchEmbeddingCV/<RUN_NAME>/<class>/; use a new name per run.
    RUN_NAME = 'R3_softmax_Method2'

    target_cols = [class_fraction_col(c) for c in CLASS_NAMES]
    tables = []
    for rabbit_id, block_num in RABBIT_BLOCKS:
        t = load_rabbit_table(rabbit_id, block_num, BASE_DIR, PATCH_CSV_NAME, target_cols,
                              min_he_coverage=MIN_HE_COVERAGE, min_mr_coverage=MIN_MR_COVERAGE)
        if t is not None:
            tables.append(t)

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

    run_dir = Path(__file__).parent / 'TestData' / 'PatchEmbeddingCV' / RUN_NAME
    for class_name, target_col in zip(CLASS_NAMES, target_cols):
        target_tables = select_target(tables, target_col)
        intensity_tables = select_stage_predictors(target_tables, 'stage0')

        print(f"\n--- Per-block correlations of patch-mean MR intensity with {target_col} ---")
        corr = per_block_correlations(intensity_tables)
        print_correlation_tables(corr, value=CORR_VALUE)
        if SHOW_CORR_PLOTS:
            plot_correlation_heatmap(corr, value=CORR_VALUE, target_label=target_col)
            plt.show()

        if not RUN_MODELS:
            continue
        out_dir = str(run_dir / class_name)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, f'{CV_MODE}_metrics.csv')
        predictions_path = os.path.join(out_dir, f'{CV_MODE}_predictions.csv')
        frames = []
        # Every stage shares the same patches, so predictions are added as columns.
        predictions = patch_target_frame(target_tables, CV_MODE)

        def run_stage(label, stage_tables, model_builder):
            print(f"\n=== {target_col}: {label} ({len(stage_tables[0].predictor_names)} predictor(s)) ===")
            results = run_model_comparison(holdout_folds(stage_tables, CV_MODE), MODEL_TYPES,
                                           model_builder=model_builder)
            frames.append(results_to_frame(results, label).assign(target=target_col))
            pd.concat(frames, ignore_index=True).to_csv(metrics_path, index=False)
            for col, values in held_out_predictions(stage_tables, results, label).items():
                predictions[col] = values
            predictions.to_csv(predictions_path, index=False, float_format='%.5g')
            print(f"Saved {CV_MODE.upper()} metrics to {metrics_path} and predictions to {predictions_path}")

        # Baseline on the same rows: patch-mean intensity of each contrast, no embeddings.
        run_stage('stage0', intensity_tables, build_model)
        for stage in ('stage2', 'stage0_2'):
            stage_tables = select_stage_predictors(target_tables, stage)
            names = stage_tables[0].predictor_names
            for n_components in PCA_COMPONENTS:
                run_stage(f'{stage}_pca{n_components}', stage_tables,
                          lambda model_type: embedding_pca_model(model_type, names, n_components))

        print(f"\n=== {target_col}: {CV_MODE.upper()} summary, mean over held-out folds ===")
        with pd.option_context('display.width', 200):
            print(summarize_loro(pd.concat(frames, ignore_index=True)).round(3))
            print("\n=== Prediction level per held-out fold, patch-mean intensity baseline ===")
            print(prediction_level_summary(
                predictions, [f'stage0_{model_type}' for model_type in MODEL_TYPES]).round(3).T)
