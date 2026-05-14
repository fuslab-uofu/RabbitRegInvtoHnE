import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

DATA_DIR = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/Analysis/Block07"
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



