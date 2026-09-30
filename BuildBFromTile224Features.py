import json
import sys
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import PyQt5.sip
sys.modules.setdefault('sip', PyQt5.sip)  # landmark .npy files pickle QPointF objects, whose
                                            # unpickler does a bare `import sip`, not `PyQt5.sip`
import skimage as ski

from TileUtils import load_landmarks, get_bf_slice_index, CSV_CZI_lookup
from NewtonAdjointInit import load_moving_vox, load_source_geometry, _trilinear_stencil, normalized_adjoint_init

# Builds b/Q for NewtonAdjointInit.py's trilinear pipeline directly from the new 224px
# H&E-native tile grid's QuPath cell-count CSVs (see Tile_Annotation_Tool/TileNormalizedSlides.py
# + project_qupath_headless_celldetect_tile224px), deliberately bypassing the voxel-unique-ID/
# ComposedReg chain that PaintHnEFeatureToInVivo.build_b_and_Q_from_feature depends on -- routing
# around that chain (whose voxel outlines carry real registration error) is exactly why the
# 224px-native tiling exists.
#
# Design: PULL, not push. For every BlockFace-grid pixel, walk the same per-slide TPS spline
# TilestoVoxels.py/TilestoVoxelsV2.py already build (BlockFace -> full-res H&E) forward, land in
# the NormalizedFull TIFF's own pixel space (the same space TileNormalizedSlides.py tiled), and
# bilinearly blend the H&E tile grid at that continuous point (see bilinear_tile_pull) -- not a
# nearest-neighbor single-tile snap, since one native BlockFace voxel is roughly 1/(2.8^2) the
# area of one 224px tile and a hard snap would needlessly throw away that resolution gap.
#
# Defect handling: HARD ADMISSION THRESHOLD on calibrated clean-probability, applied at the H&E
# tile level before any interpolation touches the tile's value -- not a continuous Q weight.
# Decided after working through why continuous Q can't do this job: the fold/debris artifact is
# a systematic physical bias (more tissue thickness -> more absorbance), not zero-mean noise, and
# a weighted average with nonzero weight on a biased value dilutes the bias but can't eliminate
# it, no matter which pipeline stage the weight is applied at (tile blend, adjoint scatter, or
# solver objective). Classifier confidence is also epistemic uncertainty about a binary state,
# not a physical partial-mixture fraction -- unlike genuine geometric partial tile coverage,
# which *is* principled to blend continuously. Only a hard admit/exclude cutoff actually keeps a
# biased tile's value out of any downstream combination.
#
# Q ("On Geometric and Algebraic Inversion of Deformed Images", Sarang Joshi, 2026-09-10):
# reverts to its textbook meaning once defect-handling happens upstream -- pure geometric
# coverage of whichever tile set (filtered or not) was admitted into b, consumed downstream by
# the adjoint/reconstruction machinery (weighted coverage A*_h(Q*1), normalized adjoint init
# A*_h(Qb)/A*_h(Q1), objective 0.5||Q^(1/2)(A_h a - b)||^2) -- never folded into the
# interpolation weights themselves, which stay purely geometric (paper Eq. 12-14).

TILE_SIZE = 224   # must match Tile_Annotation_Tool/TileNormalizedSlides.py's TILE_SIZE
SCALE_FAC = 20     # full-res H&E pixels per landmark/reg.png pixel -- same constant
                   # TilestoVoxels.py/TilestoVoxelsV2.py already hardcode

# Defect/clean predictions come from HnE_Classification's bulk classifier (see
# predict_from_cache.py), read directly from its centralized output -- not copied
# per-rabbit into each Analysis folder. This is the single source for all slides:
# the 13 hand-annotated slides only cover a sample of each slide's tiles (not
# comprehensive), so there's no reason to special-case them over the model's own
# (uniformly comprehensive) predictions.
DEFECT_MODEL_NAME = 'defect_clean_logreg_pca20'


def threshold_tag(threshold):
    return 'noFilter' if threshold is None else f'thresh{threshold}'


def load_tile_feature_lookup(csv_path, feature_col):
    """tile_id -> feature_col value, as a pandas Series indexed by tile_id -- the
    vectorized equivalent of paint_feature_volume's LUT-array trick, but keyed by an
    arbitrary large tile_id instead of a compact 1..N voxel-ID range."""
    df = pd.read_csv(csv_path)
    tile_ids = df['tile_name'].str.extract(r'(\d+)$')[0].astype(np.int64)
    return pd.Series(df[feature_col].to_numpy(), index=tile_ids)


def load_clean_prevalence(metrics_path):
    """True population prevalence of 'clean' (not the 50/50 class_weight='balanced'
    trained against), read from the model's own saved metrics.json support counts --
    self-updating if the model is ever retrained on a different label set."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    clean_n = metrics['per_class']['clean']['support']
    defect_n = metrics['per_class']['defect']['support']
    return clean_n / (clean_n + defect_n)


def calibrate_balanced_proba(p_raw, pi_true):
    """Corrects a probability from a class_weight='balanced'-trained binary
    classifier back to the true population prevalence pi_true.

    class_weight='balanced' reweights the training loss so the fit behaves as if
    the two classes were exactly 50/50 -- for logistic regression specifically,
    predict_proba's output is then calibrated to that *implied* 50/50 prior, not
    the real one. This is the standard closed-form prior-correction (King & Zeng
    2001, rare-events logistic regression), collapsed to its simple form since the
    effective training-time prior here is exactly 0.5:

        p_corrected = (pi * p_raw) / (pi * p_raw + (1 - pi) * (1 - p_raw))

    No retraining needed -- this is a one-line post-hoc transform on predict_proba's
    output. Assumes logreg + PCA(20) is otherwise well-calibrated (a safe assumption
    for plain logistic regression, unlike e.g. random forests) -- i.e. this fixes
    the *known* class-reweighting distortion, not any other miscalibration source.
    """
    p_raw = np.clip(p_raw, 1e-9, 1 - 1e-9)
    odds = p_raw / (1 - p_raw) * (pi_true / (1 - pi_true))
    return odds / (1 + odds)


def load_clean_q_lookup(pred_csv_path, pi_clean):
    """tile_id -> calibrated P(clean), as a pandas Series -- pulled straight from
    predict_from_cache.py's bulk output (x0,y0,x1,y1,clean,defect -- no tile_id
    column, so it's derived here the same way bilinear_tile_pull derives tile_ids
    for pixels). This is the admission-threshold criterion, not a sampling weight."""
    df = pd.read_csv(pred_csv_path)
    col = np.floor(df['x0'].to_numpy() / TILE_SIZE).astype(np.int64)
    row = np.floor(df['y0'].to_numpy() / TILE_SIZE).astype(np.int64)
    tile_ids = (row + 10000) * 100000 + (col + 10000)
    q = calibrate_balanced_proba(df['clean'].to_numpy(), pi_clean)
    return pd.Series(q, index=tile_ids)


def bilinear_tile_pull(x, y, tile_size, lookup):
    """Bilinear interpolation across the H&E tile grid's tile-center lattice --
    replaces a nearest-neighbor single-tile snap, treating each tile's scalar
    value as a sample at its center and blending the 4 surrounding tiles by the
    continuous point's fractional position between them (same structure as the
    paper's trilinear gather, Eq. 12-14, applied to the tile grid instead of a
    raw voxel grid).

    Missing corner tiles (off-tissue, edge of slide, or excluded by a defect
    admission threshold) are excluded from the blend and the remaining weights
    renormalized -- a partial-coverage average, not zero-fill. Returns
    (value, weight_total): value is NaN where weight_total is 0; weight_total in
    [0, 1] is itself a meaningful geometric coverage confidence (1.0 with all 4
    corners covered, partial near tissue edges or a filtered-out neighbor, 0 with
    none) -- used directly as Q, whatever tile set (filtered or not) was passed in.
    """
    tx = x / tile_size - 0.5
    ty = y / tile_size - 0.5
    col0 = np.floor(tx).astype(np.int64)
    row0 = np.floor(ty).astype(np.int64)
    fx = tx - col0
    fy = ty - row0

    weighted_sum = np.zeros(len(x), dtype=np.float64)
    weight_total = np.zeros(len(x), dtype=np.float64)
    for d_col, d_row, wx, wy in (
        (0, 0, 1 - fx, 1 - fy),
        (1, 0, fx, 1 - fy),
        (0, 1, 1 - fx, fy),
        (1, 1, fx, fy),
    ):
        w = wx * wy
        tile_ids = (row0 + d_row + 10000) * 100000 + (col0 + d_col + 10000)
        vals = lookup.reindex(tile_ids).to_numpy()
        covered = np.isfinite(vals)
        weighted_sum[covered] += w[covered] * vals[covered]
        weight_total[covered] += w[covered]

    value = np.full(len(x), np.nan)
    has_coverage = weight_total > 0
    value[has_coverage] = weighted_sum[has_coverage] / weight_total[has_coverage]
    return value, weight_total


def build_b_and_Q_from_tile_features(RabbitID, Block, RabbitFolder, feature_col, thresholds=(None,),
                                      tile_size=TILE_SIZE, scale_fac=SCALE_FAC,
                                      ceph_base='/System/Volumes/Data/ceph/hifu'):
    """{threshold: (b, Q, target_affine)}, one entry per requested admission threshold.
    `None` means no defect filtering at all (baseline); a float in [0,1] means only
    tiles with calibrated P(clean) >= threshold are admitted into b -- everything else
    is treated exactly like an off-tissue tile (excluded from the bilinear blend, not
    down-weighted). Q is always pure geometric bilinear coverage of whichever tile set
    was admitted -- see module docstring for why Q isn't a continuous defect weight.

    Shares the expensive per-slide work (landmark load, TPS spline fit, the per-pixel
    coordinate pull through it) across every requested threshold -- only the cheap
    bilinear pull itself is redone per threshold, since filtering changes which tiles
    are admitted.

    Also returns (corners, source_shape, source_affine): the trilinear-stencil and
    MR-grid geometry needed for NewtonAdjointInit.normalized_adjoint_init, computed
    once here since it reuses the same `h` this function already loads -- avoids a
    second, expensive load_moving_vox/ComposedReg call in __main__.
    """
    need_predictions = any(t is not None for t in thresholds)

    Block_str = f'Block{Block:02d}'
    rabbase = os.path.join(RabbitFolder, RabbitID)
    hne_base_dir = os.path.join(rabbase, 'HnE', Block_str)
    reg_HnE_dir = os.path.join(hne_base_dir, 'Registered')
    bf_cropped_dir = os.path.join(rabbase, 'BlockFace_RGB', Block_str, 'CroppedImages')
    analysis_dir = os.path.join(rabbase, 'Analysis', Block_str)

    pi_clean = None
    defect_predictions_dir = None
    if need_predictions:
        hne_classification_dir = os.path.join(ceph_base, 'users/jbonaventura/RabbitRegistrationProj/HnE_Classification')
        defect_predictions_dir = os.path.join(hne_classification_dir, 'Annotations', 'tiff_224_defect_predictions')
        defect_metrics_path = os.path.join(hne_classification_dir, 'trainedmodels', DEFECT_MODEL_NAME, 'metrics.json')
        pi_clean = load_clean_prevalence(defect_metrics_path)
        print(f"Admission threshold(s) {[t for t in thresholds if t is not None]} against "
              f"calibrated P(clean) (pi_clean={pi_clean:.3f}, from {defect_metrics_path})")

    print("Loading h (moving_vox) / target grid...")
    h, target_affine = load_moving_vox(RabbitID, Block, RabbitFolder, return_target_affine=True)
    X, Y, Z = h.shape[:3]

    print("Loading source (MR) grid geometry + trilinear stencil (for normalized_adjoint_init)...")
    source_shape, source_affine = load_source_geometry(RabbitID, Block, RabbitFolder)
    corners = _trilinear_stencil(h, source_shape)

    arrays = {t: (np.zeros((X, Y, Z)), np.zeros((X, Y, Z))) for t in thresholds}

    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir)
                            if f.lower().endswith('reg.png') and not f.startswith('._'))

    for fname in hne_filenames:
        img_number = re.search(r'\d+', fname).group()
        t0 = time.perf_counter()

        landmarks = load_landmarks(hne_base_dir, img_number)
        src = np.array([[p.y(), p.x()] for p in landmarks[0]]) * 3  # blockface [row, col]
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]]) * 3  # H&E downsampled [x, y]
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)

        slice_num = get_bf_slice_index(bf_cropped_dir, img_number)

        CZI_filepath = CSV_CZI_lookup(RabbitID, Block_str, img_number, ceph_base=ceph_base)
        czi_stem = os.path.splitext(os.path.basename(CZI_filepath))[0]
        csv_path = os.path.join(analysis_dir, f'{czi_stem}_corrected_fullres-cellcount-tile224px.csv')
        if not os.path.exists(csv_path):
            print(f"  IMG{img_number} ({czi_stem}): no tile224px cellcount CSV yet -- skipping")
            continue
        tile_lookup = load_tile_feature_lookup(csv_path, feature_col)

        q_lookup = None
        if need_predictions:
            pred_csv_path = os.path.join(defect_predictions_dir, f'{czi_stem}_corrected_fullres_pred.csv')
            if not os.path.exists(pred_csv_path):
                raise FileNotFoundError(
                    f"{czi_stem}: no defect/clean prediction CSV at {pred_csv_path}, but this "
                    f"slide has a tile224px feature CSV -- every slide should be scored by now "
                    f"(predict_from_cache.py against all 83 slides), so a missing prediction file "
                    f"here means something upstream broke, not a normal gap to skip over."
                )
            q_lookup = load_clean_q_lookup(pred_csv_path, pi_clean)
            missing = tile_lookup.index.difference(q_lookup.index)
            if len(missing) > 0:
                raise RuntimeError(
                    f"{czi_stem}: {len(missing)} tile(s) in the feature CSV have no matching row "
                    f"in the defect/clean prediction CSV -- should be impossible given both CSVs "
                    f"are built from the same tile224px grid; something upstream is broken. "
                    f"Example missing tile_ids: {list(missing[:10])}"
                )

        # Reg.png-resolution [row, col] grid -- the same pixel space `src` landmarks were
        # picked in, so no extra scaling before calling `splines`. Only need the PNG's pixel
        # dimensions here, not its pixel data.
        with Image.open(os.path.join(reg_HnE_dir, fname)) as im:
            W_ds, H_ds = im.size  # PIL gives (width, height)

        rows, cols = np.meshgrid(np.arange(H_ds), np.arange(W_ds), indexing='ij')
        bf_pts = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float64)

        hne_xy_ds = splines(bf_pts)           # -> H&E downsampled [x, y]
        hne_xy_full = hne_xy_ds * scale_fac   # -> NormalizedFull full-res [x, y] pixel coords
        x_full, y_full = hne_xy_full[:, 0], hne_xy_full[:, 1]

        log_bits = []
        for threshold in thresholds:
            if threshold is None:
                admitted_lookup = tile_lookup
            else:
                admitted_ids = q_lookup.index[q_lookup >= threshold]
                admitted_lookup = tile_lookup[tile_lookup.index.isin(admitted_ids)]

            b_vals, b_weight = bilinear_tile_pull(x_full, y_full, tile_size, admitted_lookup)
            b_native = b_vals.reshape(H_ds, W_ds).T[::4, ::4]
            b_weight_native = b_weight.reshape(H_ds, W_ds).T[::4, ::4]
            sx, sy = b_native.shape

            valid = b_weight_native > 0
            arr_b, arr_Q = arrays[threshold]
            arr_b[:sx, :sy, slice_num] = np.where(valid, b_native, 0.0)
            arr_Q[:sx, :sy, slice_num] = np.where(valid, 1.0, 0.0)
            log_bits.append(f"{threshold_tag(threshold)} coverage={valid.mean():.1%}")

        elapsed = time.perf_counter() - t0
        print(f"  IMG{img_number} ({czi_stem}) -> slice {slice_num}: {', '.join(log_bits)} ({elapsed:.1f}s)")

    results = {}
    flip = target_affine[0, 0] > 0 and target_affine[1, 1] > 0
    for threshold, (b, Q) in arrays.items():
        if flip:
            b, Q = b[::-1, ::-1, :], Q[::-1, ::-1, :]
        results[threshold] = (b, Q, target_affine)

    return results, corners, source_shape, source_affine


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID = 'R24-240'
    Block = 3
    FEATURE_COL = 'mean_Nucleus_Hematoxylin_OD_mean'
    THRESHOLDS = (None, 0.5, 0.7, 0.8, 0.9)  # None = no filtering (baseline); floats = P(clean) admission cutoff

    results, corners, source_shape, source_affine = build_b_and_Q_from_tile_features(
        RabbitID, Block, RabbitFolder, FEATURE_COL, thresholds=THRESHOLDS)

    out_dir = str(Path(__file__).parent / 'TestData' / f'{RabbitID}_Block{Block:02d}_Tile224Features')
    os.makedirs(out_dir, exist_ok=True)

    for threshold, (b, Q, target_affine) in results.items():
        tag = threshold_tag(threshold)

        # run2: b-only, for a quick look at the retrained (18-slide) defect classifier's
        # effect on b -- Q and a0init skipped (not written, and a0init not even computed)
        # since they're not needed for this comparison and a0init in particular is a
        # separate non-trivial call.
        b_path = os.path.join(out_dir, f'{RabbitID}_Block{Block:02d}_b_tile224_{FEATURE_COL}_{tag}_run2.nii.gz')
        nib.save(nib.Nifti1Image(b.astype(np.float32), target_affine), b_path)
        print(f"[{tag}] Saved b -> {b_path}")
        print(f"[{tag}] Overall coverage: {Q.mean():.2%} of the full target grid")
