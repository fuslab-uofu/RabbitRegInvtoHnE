import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from SADMS import relaxation

DATA_DIR     = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/Analysis/Block07"
MR_NAMES_CSV = "/Users/jbonaventura/Desktop/MR_renames.csv"  # path to CSV with columns: variable_name, display_name
N_GEOM = 2
#Number of MR contrast Mechanisms-
N_MR   = 12
DIRECTION = "MR_to_HnE"  # "HnE_to_MR" or "MR_to_HnE"
DAY3_MR_ONLY = True       # True: use only MR features whose display name contains "Day 3"
HNE_FEATURE_OF_INTEREST = "mean_Nucleus_Hematoxylin_OD_mean"  # column from HnE_names — spatial map, and MR_to_HnE target

# Standardized display ranges for relaxometry maps (ms), per ISMRM qMR study group
# recommendation (Fuderer et al. 2025) — same range used for all T1 maps and all T2
# maps so a given relaxation value always renders as the same color.
T1_RANGE = (0, 3000)
T2_RANGE = (0, 200)

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


def load_slide(cellcount_path):
    basename = os.path.basename(cellcount_path).replace("-cellcount.csv", "")
    texture_matches = glob.glob(os.path.join(DATA_DIR, f"voxel_features_{basename}*.csv"))
    if not texture_matches:
        print(f"No texture file found for {basename}, skipping.")
        return None
    cell_df = pd.read_csv(cellcount_path)
    texture_df = pd.read_csv(texture_matches[0])
    cell_df['poly_id'] = cell_df['tile_name'].str.extract(r'(\d+)$').astype(int)
    merged_df = texture_df.merge(cell_df, on='poly_id')
    feature_names = merged_df.drop(columns=['tile_name', 'poly_id']).columns.to_numpy()
    features = merged_df.drop(columns=['tile_name', 'poly_id']).to_numpy(dtype=float)
    return features, feature_names, basename


def compute_cross_corr(features):
    MR  = features[:, N_GEOM:N_GEOM + N_MR]
    HnE = features[:, N_GEOM + N_MR:]
    valid = ~np.isnan(features).any(axis=1)
    corr, pval = spearmanr(np.hstack([MR[valid], HnE[valid]]))
    return corr[:N_MR, N_MR:], pval[:N_MR, N_MR:]


def plot_hne_feature_spatial(slides, feature_name, feature_names, n_geom, cmap='viridis'):
    feat_idx = np.where(feature_names == feature_name)[0]
    if len(feat_idx) == 0:
        raise ValueError(f"Feature '{feature_name}' not found in feature_names")
    feat_idx = feat_idx[0]

    all_vals = np.concatenate([s[0][:, feat_idx] for s in slides])
    vmin, vmax = np.nanpercentile(all_vals, [1, 99])

    n_slides = len(slides)
    ncols = min(3, n_slides)
    nrows = int(np.ceil(n_slides / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False,
                             layout='constrained')
    axes = axes.flatten()

    for ax, (features, _, basename) in zip(axes, slides):
        xy = features[:, :n_geom]
        vals = features[:, feat_idx]
        valid = ~np.isnan(vals)
        sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=vals[valid], s=12,
                        cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(basename, fontsize=9)
        ax.axis('off')

    for ax in axes[n_slides:]:
        ax.set_visible(False)

    fig.suptitle(f'{feature_name} at voxel locations', fontsize=12)
    fig.colorbar(sc, ax=axes[:n_slides].tolist(), label=feature_name, shrink=0.4)
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


def plot_fold_comparison(fold_i, oof, geom, target_names, slide_name, target_styles=None, group_size=4):
    xy = geom[fold_i]
    n_targets = len(target_names)

    for g_start in range(0, n_targets, group_size):
        group = list(range(g_start, min(g_start + group_size, n_targets)))
        fig, axes = plt.subplots(len(group), 3, figsize=(18, 4 * len(group)),
                                 squeeze=False)
        for row, target_j in enumerate(group):
            y_true, y_pred = oof[target_j][fold_i]
            err   = np.abs(y_pred - y_true) / np.abs(y_true) * 100
            style = target_styles[target_j] if target_styles is not None else {'kind': 'cmap', 'cmap': 'viridis'}

            if style['kind'] == 'relaxation':
                lo, hi = style['loLev'], style['upLev']
                true_vals, lut = relaxation(style['maptype'], y_true, lo, hi)
                pred_vals, _   = relaxation(style['maptype'], y_pred, lo, hi)
                value_cmap  = ListedColormap(lut)
                value_label = f"{style['maptype']} (ms)"
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
                ax = axes[row, col]
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=10,
                                cmap=cmap, vmin=vlo, vmax=vhi)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axis('off')
                ax.set_title(f'{target_names[target_j]} — {title}' if col == 0 else title,
                             fontsize=9)
                plt.colorbar(sc, ax=ax, label=clabel, fraction=0.046, pad=0.04)

        fig.suptitle(f'Fold {fold_i + 1} — {slide_name}  '
                     f'(Target {g_start + 1}–{min(g_start + group_size, n_targets)} of {n_targets})',
                     fontsize=12)
        plt.tight_layout()
        plt.show()


def run_rf_loocv(slides, predictor_idx, target_idx, mr_idx, predictor_names=None, target_names=None, target_styles=None):
    n_slides     = len(slides)
    n_predictors = len(predictor_idx)
    n_targets    = len(target_idx)
    r2_scores   = np.full((n_slides, n_targets), np.nan)
    importances = np.zeros((n_slides, n_targets, n_predictors))
    # out-of-fold predictions: list of length n_targets, each a list of (y_true, y_pred) per fold
    oof  = [[[] for _ in range(n_slides)] for _ in range(n_targets)]
    geom = [None] * n_slides  # (x, y) centroid coords for each fold's test set

    for fold_i in range(n_slides):
        test_feat  = slides[fold_i][0]
        train_feat = np.vstack([slides[j][0] for j in range(n_slides) if j != fold_i])
        train_mask = (~np.isnan(train_feat).any(axis=1) &
                      ~(train_feat[:, mr_idx] == 0).all(axis=1))
        test_mask  = (~np.isnan(test_feat).any(axis=1) &
                      ~(test_feat[:,  mr_idx] == 0).all(axis=1))
        X_train = train_feat[train_mask][:, predictor_idx]
        X_test  = test_feat[test_mask][:, predictor_idx]
        geom[fold_i] = test_feat[test_mask, :N_GEOM]  # (n_test_voxels, 2)
        print(f"Fold {fold_i + 1}/{n_slides} — held out: {slides[fold_i][2]}, "
              f"train={train_mask.sum()} voxels, test={test_mask.sum()} voxels")
        for target_j in range(n_targets):
            y_train = train_feat[train_mask, target_idx[target_j]]
            y_test  = test_feat[test_mask,   target_idx[target_j]]
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            r2_scores[fold_i, target_j]   = r2_score(y_test, y_pred)
            importances[fold_i, target_j] = rf.feature_importances_
            oof[target_j][fold_i] = (y_test, y_pred)
            print(f"  Target {target_j + 1}/{n_targets} done  R²={r2_scores[fold_i, target_j]:.3f}")

        if target_names is not None:
            plot_fold_comparison(fold_i, oof, geom, target_names, slides[fold_i][2], target_styles=target_styles)
        if predictor_names is not None and target_names is not None:
            plot_fold_importances(importances[fold_i:fold_i + 1], target_names, predictor_names,
                                  [slides[fold_i][2]])

    return r2_scores, importances, oof, geom
    # r2_scores:   (n_slides, n_targets)
    # importances: (n_slides, n_targets, n_predictors)
    # oof:         [n_targets][n_slides] -> (y_true, y_pred) arrays
    # geom:        [n_slides] -> (n_test_voxels, 2) centroid coords


def plot_r2(r2_scores, target_names):
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
    ax.set_title('RF regression: R² per target feature (leave-one-slide-out)')
    plt.tight_layout()
    plt.show()


def plot_best_predicted_vs_actual(r2_scores, oof, target_names, slide_names):
    best_target = int(r2_scores.mean(axis=0).argmax())
    mean_r2 = r2_scores[:, best_target].mean()
    colors = plt.cm.tab10(np.linspace(0, 1, len(slide_names)))

    fig, ax = plt.subplots(figsize=(7, 7))
    all_vals = []
    for fold_i, (y_true, y_pred) in enumerate(oof[best_target]):
        ax.scatter(y_true, y_pred, s=4, alpha=0.4, color=colors[fold_i], label=slide_names[fold_i])
        all_vals.extend([y_true.min(), y_true.max()])

    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel(f'Actual {target_names[best_target]}')
    ax.set_ylabel(f'Predicted {target_names[best_target]}')
    ax.set_title(f'Predicted vs actual — {target_names[best_target]}  (mean R²={mean_r2:.3f})')
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_spatial_accuracy(r2_scores, oof, geom, target_names, slide_names):
    best_target = int(r2_scores.mean(axis=0).argmax())
    n_slides = len(slide_names)
    ncols = 3
    nrows = int(np.ceil(n_slides / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for fold_i, (y_true, y_pred) in enumerate(oof[best_target]):
        ax = axes[fold_i]
        xy  = geom[fold_i]
        err = np.abs(y_pred - y_true) / np.abs(y_true) * 100
        sc  = ax.scatter(xy[:, 0], xy[:, 1], c=err, s=15, cmap=ERR_CMAP, vmin=ERR_VMIN, vmax=ERR_VMAX)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(slide_names[fold_i], fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(sc, ax=ax, label='% error')

    for ax in axes[n_slides:]:
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


def plot_fold_importances(importances, target_names, predictor_names, slide_names, top_n=10):
    n_slides, n_targets, n_predictors = importances.shape
    top_n = min(top_n, n_predictors)
    ncols = 4
    nrows = int(np.ceil(n_targets / ncols))
    for fold_i in range(n_slides):
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
        fig.suptitle(f'Fold {fold_i + 1} — held out: {slide_names[fold_i]}\n'
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


# --- Load all slides ---
cellcount_files = sorted(glob.glob(os.path.join(DATA_DIR, "HnE_*-cellcount.csv")))
slides = [load_slide(f) for f in cellcount_files]
slides = [s for s in slides if s is not None]

feature_names = slides[0][1]
MR_names  = feature_names[N_GEOM:N_GEOM + N_MR]
HnE_names = feature_names[N_GEOM + N_MR:]

MR_styles = [get_mr_style(n) for n in MR_names]

if MR_NAMES_CSV:
    rename_df = pd.read_csv(MR_NAMES_CSV, header=None)
    name_map  = dict(zip(rename_df.iloc[0], rename_df.iloc[1]))
    MR_names  = np.array([name_map.get(n, n) for n in MR_names])

# --- Spatial map of a single HnE feature ---
plot_hne_feature_spatial(slides, HNE_FEATURE_OF_INTEREST, feature_names, N_GEOM)

# --- Per-slide correlation figure ---
fig, axes = plt.subplots(len(slides), 1, figsize=(18, 5 * len(slides)))
for ax, (features, _, basename) in zip(axes, slides):
    cross_corr, _ = compute_cross_corr(features)
    im = plot_corr(ax, cross_corr, MR_names, HnE_names, basename)
    plt.colorbar(im, ax=ax, label='Spearman r')
plt.tight_layout()
plt.show()

# --- Pooled correlation figure ---
all_features = np.vstack([s[0] for s in slides])
cross_corr_pooled, _ = compute_cross_corr(all_features)

fig, ax = plt.subplots(figsize=(18, 5))
im = plot_corr(ax, cross_corr_pooled, MR_names, HnE_names, "All slides pooled")
plt.colorbar(im, ax=ax, label='Spearman r')
plt.tight_layout()
plt.show()


# --- Random forest regression (leave-one-slide-out) ---
mr_idx  = np.arange(N_GEOM, N_GEOM + N_MR)
hne_idx = np.arange(N_GEOM + N_MR, len(feature_names))

if DAY3_MR_ONLY:
    day3_mask = np.array(['Day 3' in n for n in MR_names])
    mr_idx    = mr_idx[day3_mask]
    MR_names  = MR_names[day3_mask]
    MR_styles = [s for s, keep in zip(MR_styles, day3_mask) if keep]

if DIRECTION == "HnE_to_MR":
    predictor_idx, target_idx = hne_idx, mr_idx
    predictor_names, target_names = HnE_names, MR_names
    target_styles = MR_styles
elif DIRECTION == "MR_to_HnE":
    matches = np.where(feature_names == HNE_FEATURE_OF_INTEREST)[0]
    if len(matches) == 0:
        raise ValueError(f"Feature '{HNE_FEATURE_OF_INTEREST}' not found in feature_names")
    predictor_idx, target_idx = mr_idx, matches
    predictor_names, target_names = MR_names, np.array([HNE_FEATURE_OF_INTEREST])
    target_styles = None
else:
    raise ValueError(f"Unknown DIRECTION: {DIRECTION!r}")

r2_scores, importances, oof, geom = run_rf_loocv(slides, predictor_idx, target_idx, mr_idx,
                                                   predictor_names=predictor_names, target_names=target_names,
                                                   target_styles=target_styles)
slide_names = [s[2] for s in slides]
plot_r2(r2_scores, target_names)
plot_best_predicted_vs_actual(r2_scores, oof, target_names, slide_names)
plot_spatial_accuracy(r2_scores, oof, geom, target_names, slide_names)
plot_importances(importances, target_names, predictor_names)

# --- HnE self-correlation (pooled) ---
HnE_all = all_features[:, N_GEOM + N_MR:]
valid = ~np.isnan(all_features).any(axis=1)
hne_corr, _ = spearmanr(HnE_all[valid])

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(hne_corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(HnE_names)))
ax.set_xticklabels(HnE_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(HnE_names)))
ax.set_yticklabels(HnE_names, fontsize=7)
plt.colorbar(im, ax=ax, label='Spearman r')
ax.set_title("HnE feature self-correlation (pooled)")
plt.tight_layout()
plt.show()

# --- HnE feature variance ---
plot_hne_variance(HnE_all, HnE_names, valid)


