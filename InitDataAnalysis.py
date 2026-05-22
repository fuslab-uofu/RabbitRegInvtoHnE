import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA_DIR     = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/Analysis/Block07"
MR_NAMES_CSV = "/Users/jbonaventura/Desktop/MR_renames.csv"  # path to CSV with columns: variable_name, display_name
N_GEOM = 2
N_MR   = 10


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


def plot_fold_comparison(fold_i, oof, geom, MR_names, slide_name, group_size=4):
    xy = geom[fold_i]
    n_mr = len(MR_names)

    for g_start in range(0, n_mr, group_size):
        group = list(range(g_start, min(g_start + group_size, n_mr)))
        fig, axes = plt.subplots(len(group), 3, figsize=(18, 4 * len(group)),
                                 squeeze=False)
        for row, mr_j in enumerate(group):
            y_true, y_pred = oof[mr_j][fold_i]
            vmin  = min(y_true.min(), y_pred.min())
            vmax  = max(y_true.max(), y_pred.max())
            err   = np.abs(y_pred - y_true) / np.abs(y_true) * 100
            emax  = np.nanpercentile(err, 95)

            for col, (vals, title, cmap, lo, hi, clabel) in enumerate([
                (y_true, 'Actual',    'viridis', vmin, vmax,  ''),
                (y_pred, 'Predicted', 'viridis', vmin, vmax,  ''),
                (err,    '% error',   'hot_r',   0,    emax,  '% error'),
            ]):
                ax = axes[row, col]
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=10,
                                cmap=cmap, vmin=lo, vmax=hi)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axis('off')
                ax.set_title(f'{MR_names[mr_j]} — {title}' if col == 0 else title,
                             fontsize=9)
                plt.colorbar(sc, ax=ax, label=clabel, fraction=0.046, pad=0.04)

        fig.suptitle(f'Fold {fold_i + 1} — {slide_name}  '
                     f'(MR {g_start + 1}–{min(g_start + group_size, n_mr)} of {n_mr})',
                     fontsize=12)
        plt.tight_layout()
        plt.show()


def run_rf_loocv(slides, n_geom, n_mr, mr_names=None):
    n_slides = len(slides)
    n_hne = slides[0][0].shape[1] - n_geom - n_mr
    r2_scores   = np.full((n_slides, n_mr), np.nan)
    importances = np.zeros((n_slides, n_mr, n_hne))
    # out-of-fold predictions: list of length n_mr, each a list of (y_true, y_pred) per fold
    oof  = [[[] for _ in range(n_slides)] for _ in range(n_mr)]
    geom = [None] * n_slides  # (x, y) centroid coords for each fold's test set

    for fold_i in range(n_slides):
        test_feat  = slides[fold_i][0]
        train_feat = np.vstack([slides[j][0] for j in range(n_slides) if j != fold_i])
        train_mask = ~np.isnan(train_feat).any(axis=1)
        test_mask  = ~np.isnan(test_feat).any(axis=1)
        X_train = train_feat[train_mask, n_geom + n_mr:]
        X_test  = test_feat[test_mask,   n_geom + n_mr:]
        geom[fold_i] = test_feat[test_mask, :n_geom]  # (n_test_voxels, 2)
        print(f"Fold {fold_i + 1}/{n_slides} — held out: {slides[fold_i][2]}, "
              f"train={train_mask.sum()} voxels, test={test_mask.sum()} voxels")
        for mr_j in range(n_mr):
            y_train = train_feat[train_mask, n_geom + mr_j]
            y_test  = test_feat[test_mask,   n_geom + mr_j]
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            r2_scores[fold_i, mr_j]   = r2_score(y_test, y_pred)
            importances[fold_i, mr_j] = rf.feature_importances_
            oof[mr_j][fold_i] = (y_test, y_pred)
            print(f"  MR {mr_j + 1}/{n_mr} done  R²={r2_scores[fold_i, mr_j]:.3f}")

        if mr_names is not None:
            plot_fold_comparison(fold_i, oof, geom, mr_names, slides[fold_i][2])

    return r2_scores, importances, oof, geom
    # r2_scores:   (n_slides, n_mr)
    # importances: (n_slides, n_mr, n_hne)
    # oof:         [n_mr][n_slides] -> (y_true, y_pred) arrays
    # geom:        [n_slides] -> (n_test_voxels, 2) centroid coords


def plot_r2(r2_scores, MR_names):
    means = r2_scores.mean(axis=0)
    stds  = r2_scores.std(axis=0)
    x = np.arange(len(MR_names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.7)
    for fold_r2 in r2_scores:
        ax.scatter(x, fold_r2, color='black', s=20, alpha=0.6, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(MR_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title('RF regression: R² per MR feature (leave-one-slide-out)')
    plt.tight_layout()
    plt.show()


def plot_best_predicted_vs_actual(r2_scores, oof, MR_names, slide_names):
    best_mr = int(r2_scores.mean(axis=0).argmax())
    mean_r2 = r2_scores[:, best_mr].mean()
    colors = plt.cm.tab10(np.linspace(0, 1, len(slide_names)))

    fig, ax = plt.subplots(figsize=(7, 7))
    all_vals = []
    for fold_i, (y_true, y_pred) in enumerate(oof[best_mr]):
        ax.scatter(y_true, y_pred, s=4, alpha=0.4, color=colors[fold_i], label=slide_names[fold_i])
        all_vals.extend([y_true.min(), y_true.max()])

    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel(f'Actual {MR_names[best_mr]}')
    ax.set_ylabel(f'Predicted {MR_names[best_mr]}')
    ax.set_title(f'Predicted vs actual — {MR_names[best_mr]}  (mean R²={mean_r2:.3f})')
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_spatial_accuracy(r2_scores, oof, geom, MR_names, slide_names):
    best_mr = int(r2_scores.mean(axis=0).argmax())
    n_slides = len(slide_names)
    ncols = 3
    nrows = int(np.ceil(n_slides / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    all_err = np.concatenate([np.abs(y_pred - y_true) / np.abs(y_true) * 100
                               for y_true, y_pred in oof[best_mr]])
    vmax = np.nanpercentile(all_err, 95)
    for fold_i, (y_true, y_pred) in enumerate(oof[best_mr]):
        ax = axes[fold_i]
        xy  = geom[fold_i]
        err = np.abs(y_pred - y_true) / np.abs(y_true) * 100
        sc  = ax.scatter(xy[:, 0], xy[:, 1], c=err, s=15, cmap='hot_r', vmin=0, vmax=vmax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(slide_names[fold_i], fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(sc, ax=ax, label='% error')

    for ax in axes[n_slides:]:
        ax.set_visible(False)

    fig.suptitle(f'Spatial prediction error — {MR_names[best_mr]}', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_importances(importances, MR_names, HnE_names):
    mean_imp = importances.mean(axis=0)  # (n_mr, n_hne)
    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(mean_imp, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(HnE_names)))
    ax.set_xticklabels(HnE_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(MR_names)))
    ax.set_yticklabels(MR_names, fontsize=8)
    ax.set_title('RF feature importances (mean across folds)')
    plt.colorbar(im, ax=ax, label='Importance')
    plt.tight_layout()
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

if MR_NAMES_CSV:
    rename_df = pd.read_csv(MR_NAMES_CSV, header=None)
    name_map  = dict(zip(rename_df.iloc[0], rename_df.iloc[1]))
    MR_names  = np.array([name_map.get(n, n) for n in MR_names])

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

#
# # --- Random forest regression (leave-one-slide-out) ---
# r2_scores, importances, oof, geom = run_rf_loocv(slides, N_GEOM, N_MR, mr_names=MR_names)
# slide_names = [s[2] for s in slides]
# plot_r2(r2_scores, MR_names)
# plot_best_predicted_vs_actual(r2_scores, oof, MR_names, slide_names)
# plot_spatial_accuracy(r2_scores, oof, geom, MR_names, slide_names)
# plot_importances(importances, MR_names, HnE_names)

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


