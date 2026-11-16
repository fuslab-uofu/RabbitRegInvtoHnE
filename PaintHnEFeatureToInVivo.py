import os
import glob
import tempfile
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt


def load_and_aggregate_hne_features(analysis_dir, feature_col, filename_glob='*corrected_fullres-cellcount.csv'):
    """Load every file matching filename_glob in analysis_dir, extract voxel_id
    from tile_name, and pool duplicate rows for the same voxel_id -- duplicates come
    from a voxel traced as multiple fragments within one slide, and from the same
    voxel legitimately appearing across adjacent slides (a 0.5mm MR voxel can span
    two 0.25mm-spaced physical sections), so pooling has to happen across every
    slide's CSV together, not per-file.

    feature_col is cell-count-weighted-averaged across duplicates -- a 2-cell
    fragment shouldn't count equally to a 70-cell fragment of the same voxel.
    cell_count itself is summed, since it's a total, not a per-cell statistic.
    filename_glob defaults to the per-cell QuPath measurement CSVs, but can be
    pointed at a different feature's CSV(s) (e.g. a pixelwise-intensity export
    reusing the same tile_name/cell_count/feature_col shape).
    """
    files = sorted(glob.glob(os.path.join(analysis_dir, filename_glob)))
    if not files:
        raise FileNotFoundError(f"No cellcount CSVs found in {analysis_dir}")

    # A fragment with cell_count==0 has no cells to compute feature_col over (QuPath
    # reports it as NaN) -- drop these per-file before concatenating (not after), so
    # pandas never has to reconcile an all-NaN feature_col against real float values
    # across frames. A voxel whose every fragment is zero-cell correctly ends up
    # absent from the aggregate entirely (painted as NaN/background, not 0).
    frames = []
    n_zero_cell = 0
    for f in files:
        df = pd.read_csv(f)
        df['voxel_id'] = df['tile_name'].str.extract(r'(\d+)$')[0].astype(int)
        df = df[['voxel_id', 'cell_count', feature_col]]
        n_zero_cell += int((df['cell_count'] == 0).sum())
        frames.append(df[df['cell_count'] > 0])
    all_rows = pd.concat(frames, ignore_index=True)
    print(f"Dropped {n_zero_cell:,} zero-cell fragment row(s) (no OD data to average)")

    all_rows['_weighted'] = all_rows[feature_col] * all_rows['cell_count']
    grouped = all_rows.groupby('voxel_id').agg(
        cell_count=('cell_count', 'sum'),
        _weighted_sum=('_weighted', 'sum'),
    )
    grouped[feature_col] = grouped['_weighted_sum'] / grouped['cell_count']
    grouped = grouped.drop(columns='_weighted_sum')

    n_total_rows = len(all_rows)
    n_unique = len(grouped)
    print(f"Loaded {len(files)} cellcount file(s): {n_total_rows:,} rows -> "
          f"{n_unique:,} unique voxel IDs ({n_total_rows - n_unique:,} duplicate rows "
          f"pooled via cell-count-weighted average)")
    return grouped


def load_and_aggregate_hne_intensity(b_path, composed_label_path, lookup_path):
    """Group-average a pre-computed H&E intensity volume (e.g.
    NewtonAdjointInit.py's saved `b`, on the BlockFace grid) by voxel ID, via
    the composed BlockFace-grid label volume -- the same tracing/averaging
    step load_and_aggregate_hne_features does for QuPath features, but for
    a plain intensity volume already on disk instead of per-cell CSVs. Lets
    the two pipelines be compared on the exact same per-voxel feature.

    b==0 is treated as 'no data' here -- there's no separately-saved coverage
    mask, and unlike a synthetic checkerboard's deliberate background value,
    a real H&E scan essentially never lands on an exact 0.0 pixel, so this is
    a safe proxy for real data.
    """
    b = np.asarray(nib.load(b_path).dataobj)
    label_arr = nib.load(composed_label_path).get_fdata().astype(np.int64)
    if b.shape != label_arr.shape:
        raise ValueError(f"b {b.shape} / composed label volume {label_arr.shape} shape mismatch")

    covered = b != 0
    ids_at_covered = label_arr[covered]
    vals_at_covered = b[covered]
    n_labels = len(np.load(lookup_path))
    sums = np.bincount(ids_at_covered, weights=vals_at_covered, minlength=n_labels + 1)
    counts = np.bincount(ids_at_covered, minlength=n_labels + 1)
    has_data = counts > 0
    has_data[0] = False  # id 0 = background, exclude
    ids_with_data = np.nonzero(has_data)[0]
    means = sums[ids_with_data] / counts[ids_with_data]
    return pd.Series(means, index=ids_with_data, name='intensity')


def paint_feature_volume(label_nib_path, lookup_path, feature_series):
    """Remap the voxel-ID label volume (native InVivo grid, the same one id_lookup.npy
    indexes into) through a per-ID lookup table, producing a volume of feature_series's
    values in InVivo space -- same grid the Day3End_Registered MR volumes now share.
    Voxel IDs with no entry in feature_series (no H&E coverage yet) are left as NaN,
    not 0 -- 0 is a real, meaningful feature value, so it can't double as "no data".
    Doesn't save -- the caller still needs to compute the axial slice-range summary
    from the returned array before writing it into the saved volume's header."""
    label_nib = nib.load(label_nib_path)
    label_arr = label_nib.get_fdata().astype(np.int64)
    # Same correction TilestoVoxelsV2.py already applies: this grid's affine is
    # positive-diagonal, but BlockFaceToNifti's true ground-truth affine (what the
    # real BlockFace images are saved with) is negative-diagonal with no array
    # flip -- so a positive-diagonal grid's array needs this flip to land in the
    # same real-world orientation.
    if label_nib.affine[0, 0] > 0 and label_nib.affine[1, 1] > 0:
        label_arr = label_arr[::-1, ::-1, :]

    n_labels = len(np.load(lookup_path))
    lut = np.full(n_labels + 1, np.nan, dtype=np.float32)  # index 0 = background, stays NaN

    ids = feature_series.index.to_numpy()
    out_of_range = ids[(ids < 1) | (ids > n_labels)]
    if len(out_of_range):
        raise ValueError(f"{len(out_of_range)} voxel ID(s) in feature data fall outside "
                          f"the current id_lookup range (1..{n_labels}): {out_of_range[:10]}")
    lut[ids] = feature_series.to_numpy(dtype=np.float32)

    painted = lut[label_arr]

    n_tissue = int((label_arr > 0).sum())
    n_painted = int(np.isfinite(painted[label_arr > 0]).sum())
    print(f"Painted {n_painted:,} / {n_tissue:,} tissue voxels "
          f"({100 * n_painted / n_tissue:.1f}% coverage -- the rest have no H&E data yet)")

    return painted, label_nib.affine


def build_b_and_Q_from_feature(analysis_dir, composed_label_path, lookup_path, feature_col,
                                filename_glob='*corrected_fullres-cellcount.csv'):
    """Builds b/Q for NewtonAdjointInit.py's trilinear pipeline directly from QuPath
    cell-count CSV features (e.g. mean_Nucleus_Hematoxylin_OD_mean), instead of
    build_b_and_Q's raw registered-PNG grayscale intensity. Reuses
    load_and_aggregate_hne_features (CSV -> per-voxel-ID feature, cell-count-weighted
    across duplicate fragments/slides) and paint_feature_volume (keyed paint onto the
    composed BlockFace-grid label volume) -- painting there is pixel-exact rather than
    an approximation, since every BlockFace-grid pixel already carries the voxel ID
    that owns it via the same registration the polygons themselves were traced from.

    Returns b (painted feature, NaN->0), Q (1 where painted, 0 elsewhere -- the
    coverage mask), and the composed label volume's own affine (the grid both now
    live on)."""
    feature_series = load_and_aggregate_hne_features(analysis_dir, feature_col, filename_glob=filename_glob)[feature_col]
    painted, affine = paint_feature_volume(composed_label_path, lookup_path, feature_series)
    Q = np.isfinite(painted).astype(np.float64)
    b = np.where(Q > 0, painted, 0.0).astype(np.float64)
    return b, Q, affine


def valid_mask(arr):
    """'Real signal' for the painted H&E volume -- NaN is the only no-data
    sentinel `paint_feature_volume` writes; 0 is a real feature value here
    and must not be treated as missing."""
    return np.isfinite(arr)


def compute_axial_centroid(he_arr):
    """Single (L-R, A-P) anchor point used for every axial slice's tile
    placement -- mean index of every voxel with real H&E coverage anywhere
    along the S-I (axis 1) axis, so the same anchor applies regardless of
    which slice ends up tiled."""
    coverage = valid_mask(he_arr).any(axis=1)
    xs, ys = np.nonzero(coverage)
    return xs.mean(), ys.mean()


def compute_axial_extent(he_arr):
    """(x_lo, x_hi, y_lo, y_hi) bounding box of H&E coverage in the (L-R,
    A-P) plane -- same projection as compute_axial_centroid (collapsed across
    every slice, so one fixed box applies to all of them), min/max instead of
    mean. Lets a tiling script compare tile_size against the real footprint
    of the tissue instead of the full FOV."""
    coverage = valid_mask(he_arr).any(axis=1)
    xs, ys = np.nonzero(coverage)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def compute_slice_he_fractions(he_arr):
    """Whole-slice H&E coverage fraction for every S-I (axial) slice.
    Deliberately computed over the real, in-bounds (L-R, A-P) plane only --
    no tile-window slicing here -- so there's no out-of-bounds/wraparound
    risk even though actual tiles (224 voxels) will be centered+padded past
    the A-P axis's real extent (144 voxels)."""
    n_slices = he_arr.shape[1]
    return np.array([valid_mask(he_arr[:, z, :]).mean() for z in range(n_slices)])


def build_axial_summary(he_arr, threshold):
    """Compresses the per-slice threshold pass/fail, plus the (L-R, A-P)
    coverage bounding box, into a single metadata row -- the slice range is
    valid only because the kept slices form one contiguous band; asserts that
    rather than assuming it, since a stray isolated slice at the tissue's
    edge dropping below threshold would otherwise silently vanish into the
    min/max range. Keys are kept short (cx/cy/zlo/zhi/xlo/xhi/ylo/yhi) since
    all 8 have to fit inside a NIfTI header's 80-byte descrip field."""
    cx, cy = compute_axial_centroid(he_arr)
    x_lo, x_hi, y_lo, y_hi = compute_axial_extent(he_arr)
    fractions = compute_slice_he_fractions(he_arr)
    keep = np.nonzero(fractions >= threshold)[0]

    nonzero_fracs = fractions[fractions > 0]
    print(f"H&E centroid (x,y) = ({cx:.1f}, {cy:.1f}), "
          f"extent x=[{x_lo},{x_hi}] y=[{y_lo},{y_hi}]")
    print(f"Nonzero-slice fraction stats: min={nonzero_fracs.min():.4f}, "
          f"median={np.median(nonzero_fracs):.4f}, max={nonzero_fracs.max():.4f}")
    print(f"{len(keep)}/{len(fractions)} slice(s) kept at threshold {threshold}")

    slice_start, slice_end = int(keep.min()), int(keep.max())
    if len(keep) != slice_end - slice_start + 1:
        raise ValueError(
            f"Kept slices aren't contiguous ({len(keep)} kept, but range "
            f"{slice_start}-{slice_end} spans {slice_end - slice_start + 1}) -- "
            f"a single slice-range summary would silently include slice(s) "
            f"that never passed the threshold. Inspect the fraction curve."
        )

    summary = {
        'cx': int(round(cx)),
        'cy': int(round(cy)),
        'zlo': slice_start,
        'zhi': slice_end,
        'xlo': x_lo,
        'xhi': x_hi,
        'ylo': y_lo,
        'yhi': y_hi,
    }
    return summary, fractions, (cx, cy)


def format_axial_slice_descrip(summary):
    """Encodes the slice-range summary as `key=val,key=val,...`, sized to fit
    comfortably inside a NIfTI header's 80-byte `descrip` field."""
    return ','.join(f'{k}={v}' for k, v in summary.items())


def read_axial_slice_metadata(nifti_path):
    """Reads the slice-range summary back out of a volume's `descrip` field --
    the paired reader for `format_axial_slice_descrip`, so a consumer doesn't
    have to hand-parse the header string."""
    descrip = nib.load(nifti_path).header['descrip'].item().decode()
    if '=' not in descrip:
        raise ValueError(
            f"{nifti_path}'s descrip field is empty/unrecognized ({descrip!r}) -- "
            f"this volume predates the slice-range metadata being written, or was "
            f"produced some other way. Rerun paint_feature_volume for it."
        )
    return {k: int(v) for k, v in (kv.split('=') for kv in descrip.split(','))}


def pick_representative_slices(fractions, n=8):
    """n slice indices spread evenly across every slice with any real H&E
    coverage -- deliberately spans both sides of a candidate threshold so the
    plot is useful for tuning it, not just for confirming slices already
    kept."""
    nonzero = np.nonzero(fractions > 0)[0]
    idx = np.linspace(0, len(nonzero) - 1, min(n, len(nonzero))).round().astype(int)
    return nonzero[idx]


def plot_he_fraction_curve(fractions, threshold):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fractions, marker='.', linewidth=0.8)
    ax.axhline(threshold, color='red', linestyle='--', label=f'threshold={threshold}')
    ax.set_xlabel('slice index (S-I axis)')
    ax.set_ylabel('whole-slice H&E fraction')
    ax.set_title('H&E coverage fraction per axial slice')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_representative_axial_slices(he_arr, fractions, cx, cy, zs, threshold):
    ncols = min(4, len(zs))
    nrows = int(np.ceil(len(zs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, z in zip(axes, zs):
        frac = fractions[z]
        ax.imshow(he_arr[:, z, :], cmap='gray', origin='lower')
        ax.plot(cy, cx, marker='+', color='red', markersize=12, markeredgewidth=2)
        kept = 'KEPT' if frac >= threshold else 'dropped'
        ax.set_title(f'slice {z}: frac={frac:.4f} ({kept})', fontsize=9)
        ax.axis('off')
    for ax in axes[len(zs):]:
        ax.axis('off')

    fig.suptitle('Representative axial slices, H&E fraction')
    fig.tight_layout()
    return fig


def paint_rabbit(rabbit_id, block_num, base_dir, threshold, mode='qupath', feature_col=None,
                  b_path=None, show_plots=True, filename_glob='voxel_mean_intensity_*.csv',
                  paint_space='blockface'):
    """mode='qupath': paint a per-cell QuPath feature (feature_col) aggregated from
    Analysis CSVs. mode='intensity': paint a pre-computed H&E intensity volume at
    b_path (e.g. NewtonAdjointInit.py's saved `b`), for a direct comparison against
    that pipeline's a0/a on the same feature.

    paint_space='invivo' (default) paints onto the native InVivo grid, as usual.
    paint_space='blockface' (mode='qupath' only, for now) paints onto the same
    BlockFace grid b/a0 already live on -- no InVivo round-trip -- for a direct,
    same-grid comparison against b before either pipeline's shared InVivo-painting
    step, to help isolate which side of the pipeline a mismatch actually comes from."""
    Block = f'Block{block_num:02d}'
    rabbase = os.path.join(base_dir, f'R{rabbit_id}')

    voxel_maps_dir = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps')
    label_nib_path = os.path.join(voxel_maps_dir, 'voxel_tile_vol.nii.gz')
    lookup_path    = os.path.join(voxel_maps_dir, 'voxel_tile_vol_id_lookup.npy')
    composed_label_path = os.path.join(voxel_maps_dir, f'voxel_tile_vol_ComposedRegTo{Block}.nii.gz')

    analysis_dir = os.path.join(rabbase, 'Analysis', Block)

    print(f"--- Rabbit {rabbit_id} ({mode}, {paint_space}) ---")
    if mode == 'qupath':
        aggregated = load_and_aggregate_hne_features(analysis_dir, feature_col, filename_glob=filename_glob)
        feature_series = aggregated[feature_col]
        if paint_space == 'blockface':
            paint_label_path = composed_label_path
            out_path = os.path.join(tempfile.gettempdir(),
                                     f"R{rabbit_id}_Block{block_num:02d}_{feature_col}_blockface_test.nii.gz")
        else:
            paint_label_path = label_nib_path
            out_path = os.path.join(analysis_dir, f'hne_{feature_col}_invivo.nii.gz')
    elif mode == 'intensity':
        feature_series = load_and_aggregate_hne_intensity(b_path, composed_label_path, lookup_path)
        paint_label_path = label_nib_path
        # Real destination once this is production-ready:
        # out_path = os.path.join(analysis_dir, 'hne_intensity_invivo.nii.gz')
        # For now: save alongside NewtonAdjointInit.py's a0_test/a_test/b_test outputs
        # (same tempfile.gettempdir() location) for a direct side-by-side Slicer comparison.
        out_path = os.path.join(tempfile.gettempdir(), f"R{rabbit_id}_Block{block_num:02d}_painted_test.nii.gz")
    else:
        raise ValueError(f"Unknown mode {mode!r} -- expected 'qupath' or 'intensity'")

    painted, affine = paint_feature_volume(paint_label_path, lookup_path, feature_series)
    summary, fractions, (cx, cy) = build_axial_summary(painted, threshold)

    img = nib.Nifti1Image(painted, affine)
    img.header['descrip'] = format_axial_slice_descrip(summary)
    nib.save(img, out_path)
    print(f"Saved -> {out_path}")
    print(f"Round-trip check: {read_axial_slice_metadata(out_path)}")

    if show_plots:
        plot_he_fraction_curve(fractions, threshold)
        zs = pick_representative_slices(fractions, n=8)
        plot_representative_axial_slices(painted, fractions, cx, cy, zs, threshold)


if __name__ == '__main__':
    BASE_DIR = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RABBIT_BLOCKS = {"24-240": 3}
    HE_FRACTION_THRESHOLD = 0.0005  # deliberately very inclusive
    Show_Plots = True

    MODE = 'qupath'  # 'qupath' (per-cell CSV feature) or 'intensity' (pre-computed b volume)
    FEATURE_COL = 'mean_intensity'  # mode='qupath' only

    for rabbit_id, block_num in RABBIT_BLOCKS.items():
        # mode='intensity' only: NewtonAdjointInit.py's own __main__ saves its b here.
        b_path = os.path.join(tempfile.gettempdir(), f"R{rabbit_id}_Block{block_num:02d}_b_test.nii.gz")
        print(b_path)
        paint_rabbit(rabbit_id, block_num, BASE_DIR, HE_FRACTION_THRESHOLD, mode=MODE,
                     feature_col=FEATURE_COL, b_path=b_path, show_plots=Show_Plots)

    if Show_Plots:
        plt.show()
