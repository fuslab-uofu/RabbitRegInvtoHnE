#Summary figure across every MR -> H&E technique in one RadPathPredictions run folder: one
#metric for one model (e.g. XGBoost pooled Spearman rho), per class, for each CV mode. Reads
#each <stage>_<cv>/metrics.csv the prediction scripts wrote, so it picks up whatever has run.
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RabbitPathFinder import prediction_run_dir

CV_TITLES = {'loro': 'LORO: held-out rabbit, trained on the other rabbits',
             'lobo': 'LOBO: held-out block, trained on the same rabbit\'s other blocks'}
MODEL_LABELS = {'rf': 'Random forest', 'xgboost': 'XGBoost', 'linear': 'Linear'}
METRIC_LABELS = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'pearson_r': 'Pearson r', 'spearman_rho': 'Spearman ρ'}
# Stage families in plot order: (family label, stage-name pattern, tick label from the match).
STAGE_FAMILIES = [
    ('Stage 0', r'stage0$', lambda m: 'raw MR'),
    ('Stage 1a', r'stage1a$', lambda m: 'all features'),
    ('Stage 1a', r'stage1a_(2d|3d)_r(\d+)$', lambda m: f'{m[1].upper()} r{m[2]} only'),
    ('Stage 1b', r'stage1b_mean_tile(\d+)$', lambda m: f'tile {m[1]} mean MR'),
    ('Stage 1b', r'stage1b_tile(\d+)$', lambda m: f'tile {m[1]}'),
    ('Stage 2', r'stage2_patchmean$', lambda m: 'patch mean MR'),
    ('Stage 2', r'stage2_pca(\d+)$', lambda m: f'{m[1]} PCs'),
    ('Stage 2 one grid', r'stage2_unstaggered_patchmean$', lambda m: 'patch mean MR'),
    ('Stage 2 one grid', r'stage2_unstaggered_pca(\d+)$', lambda m: f'{m[1]} PCs'),
]
# Validated categorical slots 1-3 (dataviz reference palette, all-pairs CVD-safe), plus a
# marker per class so class never depends on colour alone.
CLASS_STYLES = {'muscle': ('#2a78d6', 'o'), 'necrotic_tissue': ('#eb6834', 's'), 'immune_cells': ('#1baf7a', '^')}
CLASS_LABELS = {'muscle': 'Muscle', 'necrotic_tissue': 'Necrosis', 'immune_cells': 'Immune cells'}


def load_metrics(run_dir):
    """Every <stage>_<cv>/metrics.csv under run_dir in one frame, with 'cv' and 'class' columns
    ('class' from the target name, which MSPred and PatchEmbedding spell differently)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, '*', 'metrics.csv'))):
        cv = os.path.basename(os.path.dirname(path)).rsplit('_', 1)[1]
        frames.append(pd.read_csv(path).assign(cv=cv))
    metrics = pd.concat(frames, ignore_index=True)
    metrics['class'] = metrics['target'].str.extract(r'^(\w+?)[ _]fraction')[0]
    return metrics


def stage_order(stages):
    """[(stage, family label, tick label)] for the stages present, in STAGE_FAMILIES order
    (numbers ascending within a family)."""
    ordered = []
    for family, pattern, tick in STAGE_FAMILIES:
        matches = [(s, re.match(pattern, s)) for s in stages]
        matches = [(s, m) for s, m in matches if m]
        matches.sort(key=lambda sm: [int(g) if g.isdigit() else g for g in sm[1].groups()])
        ordered += [(s, family, tick(m)) for s, m in matches]
    unplaced = set(stages) - {s for s, _, _ in ordered}
    if unplaced:
        raise ValueError(f"No STAGE_FAMILIES entry for stage(s) {sorted(unplaced)}")
    return ordered


def plot_technique_summary(summary, metric, model, held_out):
    """One panel per CV mode: x = technique (grouped by stage family), y = metric, a dodged
    marker per class."""
    order = stage_order(summary['stage'].unique())
    x_of = {s: i for i, (s, _, _) in enumerate(order)}
    cvs = [cv for cv in CV_TITLES if cv in summary['cv'].unique()]
    classes = [c for c in CLASS_STYLES if c in summary['class'].unique()]

    fig, axes = plt.subplots(len(cvs), 1, figsize=(max(8, 0.75 * len(order) + 2), 3.6 * len(cvs) + 0.8),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    for ax, cv in zip(axes[:, 0], cvs):
        # Shaded bands mark every other stage family.
        families = [f for f in dict.fromkeys(f for _, f, _ in order)]
        for k, family in enumerate(families):
            xs = [x_of[s] for s, f, _ in order if f == family]
            if k % 2:
                ax.axvspan(min(xs) - 0.5, max(xs) + 0.5, color='#f0efec', zorder=0, linewidth=0)
        ax.axhline(0, color='#52514e', linewidth=0.8, zorder=1)
        for j, cls in enumerate(classes):
            color, marker = CLASS_STYLES[cls]
            rows = summary[(summary['cv'] == cv) & (summary['class'] == cls)]
            x = np.array([x_of[s] for s in rows['stage']]) + (j - (len(classes) - 1) / 2) * 0.22
            ax.scatter(x, rows[metric], color=color, marker=marker, s=64, edgecolor='white', linewidth=1.2,
                       zorder=3, label=CLASS_LABELS[cls])
        ax.set_title(CV_TITLES[cv], loc='left', fontsize=11)
        ax.set_ylabel(f'{held_out.capitalize()} {METRIC_LABELS.get(metric, metric)}')
        ax.grid(axis='y', color='#e4e3df', linewidth=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=0)

    bottom = axes[-1, 0]
    bottom.set_xticks(range(len(order)))
    bottom.set_xticklabels([tick for _, _, tick in order], rotation=45, ha='right')
    bottom.set_xlim(-0.6, len(order) - 0.4)
    # Family names under the tick labels, centred on each family.
    for family in dict.fromkeys(f for _, f, _ in order):
        xs = [x_of[s] for s, f, _ in order if f == family]
        bottom.annotate(family, xy=(np.mean(xs), 0), xycoords=('data', 'axes fraction'), xytext=(0, -62),
                        textcoords='offset points', ha='center', va='top', fontsize=9, color='#52514e')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(f'{MODEL_LABELS.get(model, model)} {held_out} {METRIC_LABELS.get(metric, metric)} across MR → H&E techniques',
                 x=0.01, ha='left', fontsize=13)
    return fig


if __name__ == '__main__':
    LABEL_SOURCE = 'R3_softmax'  # the run folder's H&E target source
    MR_NORM_DIR = 'Normalized_method2'  # and its MR normalization
    MODEL = 'xgboost'  # 'rf', 'xgboost' or 'linear'
    METRIC = 'spearman_rho'  # any metrics.csv column: r2, rmse, mae, pearson_r, spearman_rho
    HELD_OUT = 'pooled'  # 'pooled' = all held-out voxels/tiles/patches scored together
    SHOW_FIGURE = False

    run_dir = prediction_run_dir(LABEL_SOURCE, MR_NORM_DIR)
    out_dir = os.path.join(run_dir, 'summary_figs')
    os.makedirs(out_dir, exist_ok=True)

    metrics = load_metrics(run_dir)
    summary = metrics[(metrics['model'] == MODEL) & (metrics['held_out'] == HELD_OUT)]
    summary = summary[['cv', 'stage', 'class', METRIC]]
    name = f'{MODEL}_{HELD_OUT}_{METRIC}'
    table = summary.pivot_table(index='stage', columns=['cv', 'class'], values=METRIC)
    table = table.reindex([s for s, _, _ in stage_order(summary['stage'].unique())])
    table.to_csv(os.path.join(out_dir, f'{name}.csv'), float_format='%.3f')
    print(table.round(3).to_string())

    fig = plot_technique_summary(summary, METRIC, MODEL, HELD_OUT)
    fig_path = os.path.join(out_dir, f'{name}.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved {fig_path} and its table {name}.csv")
    plt.show() if SHOW_FIGURE else plt.close(fig)
