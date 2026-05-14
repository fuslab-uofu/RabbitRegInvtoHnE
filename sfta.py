import numpy as np
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray


def _otsu_single(counts):
    """Single Otsu threshold on a histogram. Returns 1-indexed position (matches MATLAB)."""
    counts = counts.astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, len(p) + 1))
    mu_t = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b_sq = np.where(np.isfinite(sigma_b_sq), sigma_b_sq, -np.inf)
    max_val = sigma_b_sq.max()
    if not np.isfinite(max_val):
        return 0.0
    positions = np.where(sigma_b_sq == max_val)[0] + 1  # 1-indexed to match MATLAB
    return float(np.mean(positions))


def _otsurec(I, ttotal):
    """Recursive multi-level Otsu thresholding. Returns ttotal thresholds in [0, 1]."""
    if I.size == 0:
        return np.array([])
    if I.dtype != np.uint8:
        I = (I * 255).clip(0, 255).astype(np.uint8)
    counts = np.bincount(I.ravel(), minlength=256).astype(float)
    num_bins = 256
    T = np.zeros(ttotal)

    def helper(lower_bin, upper_bin, t_lower, t_upper):
        # lower_bin, upper_bin: 1-indexed bin numbers (matches MATLAB convention)
        # t_lower, t_upper: 1-indexed threshold slot indices
        if t_upper < t_lower or lower_bin >= upper_bin:
            return
        lb = int(np.ceil(lower_bin))
        ub = int(np.ceil(upper_bin))
        sub_counts = counts[lb - 1:ub]  # 1-indexed [lb, ub] → 0-indexed [lb-1:ub]
        level = _otsu_single(sub_counts) + lower_bin
        insert_pos = int(np.ceil((t_lower + t_upper) / 2))
        T[insert_pos - 1] = level / num_bins
        helper(lower_bin, level, t_lower, insert_pos - 1)
        helper(level + 1, upper_bin, insert_pos + 1, t_upper)

    helper(1, num_bins, 1, ttotal)
    return T


def _find_borders(I):
    """Border pixels: foreground pixels with at least one background 8-neighbor.
    Pads with foreground (border_value=1) to match MATLAB's padarray(..., 1) behavior."""
    eroded = binary_erosion(I, structure=np.ones((3, 3)), border_value=1)
    return I & ~eroded


def _haus_dim(I):
    """Hausdorff fractal dimension via box counting."""
    max_dim = max(I.shape)
    if max_dim < 2:
        return 0.0
    new_dim = int(2 ** np.ceil(np.log2(max_dim)))
    I_pad = np.pad(I.astype(bool),
                   ((0, new_dim - I.shape[0]), (0, new_dim - I.shape[1])),
                   constant_values=False)

    box_counts = []
    resolutions = []
    box_size = new_dim
    while box_size >= 1:
        bp = new_dim // box_size
        # blocks[i, j, k, l] = I_pad[i*box_size+k, j*box_size+l]
        # so blocks.any(axis=(1, 3)) gives (bp, bp) occupied-box mask
        blocks = I_pad.reshape(bp, box_size, bp, box_size)
        box_counts.append(int(blocks.any(axis=(1, 3)).sum()))
        resolutions.append(1.0 / box_size)
        box_size //= 2

    log_res = np.log(np.array(resolutions, dtype=float))
    log_counts = np.log(np.array(box_counts, dtype=float))
    valid = np.isfinite(log_counts) & (np.array(box_counts) > 0)
    if valid.sum() < 2:
        return 0.0
    return float(np.polyfit(log_res[valid], log_counts[valid], 1)[0])


def sfta(I, nt=3):
    """
    SFTA texture feature extraction (Python port of Alceu Costa's MATLAB implementation).

    Parameters
    ----------
    I : np.ndarray
        Grayscale uint8 (H, W) or RGB uint8 (H, W, 3).
    nt : int
        Number of thresholds. Feature vector length = 6*nt - 3.

    Returns
    -------
    np.ndarray of shape (6*nt - 3,)

    Reference
    ---------
    Costa et al. 2012. "An Efficient Algorithm for Fractal Analysis of Textures." SIBGRAPI.
    """
    if I.dtype != np.uint8:
        I = (I * 255).clip(0, 255).astype(np.uint8)
    if I.ndim == 3:
        I = (rgb2gray(I) * 255).astype(np.uint8)

    T = _otsurec(I, nt)
    D = np.zeros(len(T) * 6 - 3)
    pos = 0

    for thresh in T:
        Ib = I > (thresh * 255)
        Ib = _find_borders(Ib)
        vals = I[Ib].astype(float)
        D[pos] = _haus_dim(Ib);      pos += 1
        D[pos] = vals.mean() if len(vals) > 0 else 0.0; pos += 1
        D[pos] = float(len(vals));   pos += 1

    T_ext = np.append(T, 1.0)
    for t in range(len(T_ext) - 2):
        Ib = (I > T_ext[t] * 255) & (I < T_ext[t + 1] * 255)
        Ib = _find_borders(Ib)
        vals = I[Ib].astype(float)
        D[pos] = _haus_dim(Ib);      pos += 1
        D[pos] = vals.mean() if len(vals) > 0 else 0.0; pos += 1
        D[pos] = float(len(vals));   pos += 1

    return D


def sfta_feature_dict(I, nt=3):
    """Returns sfta() output as a named dict, compatible with extract_features() style."""
    feats = sfta(I, nt)
    names = [f'sfta_{i}' for i in range(len(feats))]
    return dict(zip(names, feats))
