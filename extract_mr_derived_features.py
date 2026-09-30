import warnings
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter


def normalize_0_1(arr):
    """Rescale ARR to [0,1] using its own whole-array min/max, ignoring NaN --
    so a boundary-masked derived feature's own no-data positions don't
    corrupt the range. Whole-array (not per-tile, not per-slice), matching
    the same normalization discipline the raw MR 'Normalized' volumes already
    use (see extract_mri_embedding.py's own requirement: one fixed (lo, hi)
    for the whole volume). NaN positions stay NaN -- arithmetic on NaN is
    NaN, no special-casing needed; the divide does emit a spurious
    RuntimeWarning purely from touching NaN entries even though the NaN-in
    NaN-out result is exactly what's wanted (same benign quirk documented in
    MultiRabAnalysis.py's pca_by_rabbit), so it's suppressed here rather than
    left to alarm on every real run."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return (arr - lo) / (hi - lo)


def gradient_magnitude(arr, axes=(0, 2)):
    """Gradient magnitude of a 3D volume sharing the (L-R, S-I slice, A-P)
    axis convention used throughout this pipeline, differentiated only along
    AXES. Default (0,2) is the in-plane/axial gradient (matches how every
    other stage here operates per-slice); pass (0,1) for the coronal-oriented
    2D gradient, (1,2) for sagittal-oriented, or (0,1,2) for the full 3D
    gradient. All four are just different combinations of the same three
    underlying partial derivatives, so computing several variants (e.g. all
    four, as separate extra_volume columns) is cheap -- np.gradient is the
    only real cost, run once per call on the whole volume.

    MR's own no-data convention is 0 (confirmed elsewhere in this codebase,
    no NaNs) -- a naive gradient would see the artificial jump from real
    tissue to that 0 placeholder as a large, meaningless "edge" right at any
    tissue/background boundary. Fixed by only trusting a voxel's gradient
    where its neighborhood along exactly the differentiated axes (3 voxels
    wide along each axis in AXES -- everything np.gradient's central-
    difference formula could have touched there -- 1 voxel, i.e. just
    itself, along any axis not in AXES) is entirely valid; everywhere else
    the gradient is genuinely undefined, so it's set to NaN rather than a
    misleading number. That NaN is a real value like any other extra_volume
    entry -- it deliberately does NOT get excluded via the 0-means-invalid
    convention raw MR contrasts use, since extract_voxelwise_features never
    checks extra_volumes against the mask.

    Returned already as the absolute magnitude (sqrt of summed squares is
    inherently non-negative -- combining the differentiated directions this
    way already IS "the absolute value of the gradient", no separate abs()
    step needed) and rescaled to [0,1] via normalize_0_1."""
    valid = (arr != 0)
    grads = np.gradient(arr, axis=axes)
    magnitude = np.sqrt(sum(g ** 2 for g in grads))

    # 3 voxels wide along each differentiated axis, 1 (no neighbor mixing)
    # along any axis not being differentiated over.
    # mode='constant', cval=0.0: treat the true array edge as invalid too,
    # rather than scipy's default reflect/wrap, matching the same
    # no-silent-wraparound discipline extract_tile already uses.
    footprint = tuple(3 if ax in axes else 1 for ax in range(arr.ndim))
    fully_valid = minimum_filter(valid.astype(np.float32), size=footprint,
                                  mode='constant', cval=0.0) > 0
    magnitude[~fully_valid] = np.nan
    return normalize_0_1(magnitude)


def neighborhood_mean_std(arr, radius, axes=(0, 2)):
    """Masked neighborhood mean and standard deviation of ARR within RADIUS
    voxels along AXES (default (0,2): in-plane/axial, matching this
    pipeline's per-slice convention; pass (0,1,2) for a full 3D
    neighborhood, etc -- same axes convention as gradient_magnitude, and for
    the same reason: a window's real extent should only cover the axes it's
    actually meant to smooth over, 1 voxel (itself) along any other axis.

    MR's own no-data convention is 0 -- a naive scipy.ndimage.uniform_filter
    would blend real tissue values with background zeros near any boundary,
    systematically biasing every boundary-adjacent voxel's neighborhood
    stats toward zero. Fixed via a masked/normalized convolution: sum the
    valid values and separately count the valid neighbors (each via
    uniform_filter * window size, since uniform_filter itself returns an
    average, not a sum), then divide -- sum-of-valid / count-of-valid -- so
    an invalid neighbor is excluded entirely from the average rather than
    silently contributing a 0. A voxel with zero valid neighbors in its
    whole window gets NaN (genuinely undefined, not a misleading number)
    rather than falling back to some default -- deliberately no
    minimum-valid-fraction threshold yet (a barely-covered window still
    produces a number here), that's a reasonable place to add one later if
    boundary-adjacent noise turns out to matter in practice.

    Std uses the mean-of-squares-minus-square-of-mean identity (same masked
    approach), clipped at 0 before the square root to guard against a tiny
    negative value from floating-point error on a near-uniform window.

    Returns (mean, std), each already rescaled to [0,1] via normalize_0_1."""
    valid = (arr != 0)
    size = tuple(2 * radius + 1 if ax in axes else 1 for ax in range(arr.ndim))
    window_n = np.prod(size)

    # mode='constant', cval=0.0: true array edge counts as invalid too, same
    # discipline as gradient_magnitude and extract_tile.
    count = uniform_filter(valid.astype(np.float32), size=size,
                            mode='constant', cval=0.0) * window_n
    sum_x = uniform_filter(np.where(valid, arr, 0.0), size=size,
                            mode='constant', cval=0.0) * window_n
    sum_x2 = uniform_filter(np.where(valid, arr ** 2, 0.0), size=size,
                             mode='constant', cval=0.0) * window_n

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        mean = sum_x / count
        mean_sq = sum_x2 / count
    no_valid = (count == 0)
    mean[no_valid] = np.nan

    variance = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(variance)
    std[no_valid] = np.nan

    return normalize_0_1(mean), normalize_0_1(std)
