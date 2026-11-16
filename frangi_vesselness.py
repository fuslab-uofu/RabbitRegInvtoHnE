import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage._shared.utils import _supported_float_type

# 3D only (Frangi runs on the full 3D volume in this pipeline, never per-slice
# -- see project discussion on why: a curved 2D cross-section of any 3D shell
# looks tube-like regardless of the shell's real 3D shape, so only a genuine
# 3D Hessian can tell a plate/shell apart from an actual vessel).


def frangi_eigvals_by_sigma(volume, sigmas, black_ridges=False, mode='reflect', cval=0):
    """Precompute 3D Hessian eigenvalues at each sigma -- the expensive part
    of Frangi (multi-scale Hessian + eigendecomposition over the whole
    volume) -- done ONCE regardless of how many (alpha, beta, gamma)
    combinations get tried afterward via frangi_vesselness_from_eigvals.
    Mirrors skimage.filters.frangi's own internals exactly (same
    hessian_matrix/hessian_matrix_eigvals calls, same magnitude sort, same
    black_ridges negation), verified to reproduce its output bit-for-bit
    when recombined with matching alpha/beta/gamma. Returns a list of
    (3, *volume.shape) eigenvalue arrays, one per sigma, sorted by
    magnitude ascending (skimage's own convention)."""
    image = volume.astype(_supported_float_type(volume.dtype), copy=False)
    if not black_ridges:
        image = -image
    eigvals_by_sigma = []
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(
            hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True)
        )
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        eigvals_by_sigma.append(eigvals)
    return eigvals_by_sigma


def frangi_vesselness_from_eigvals(eigvals_by_sigma, alpha=0.5, beta=0.5, gamma=4.0):
    """Cheap recombination step (no Hessian recomputation) -- applies
    Frangi's (alpha, beta, gamma) formula to already-computed per-sigma
    eigenvalues and takes the max across scales. Cheap enough to be
    slider-live, unlike frangi_eigvals_by_sigma.

    alpha: sensitivity to deviation from a PLATE-like structure (r_a =
    lambda2/lambda3 -- near 0 for a plate, near 1 for a tube; larger alpha
    makes the filter stricter about excluding plate/shell-like structures,
    e.g. an ablation-cavity rim, before it counts as vessel-like).
    beta: sensitivity to deviation from a BLOB-like structure -- not the
    relevant knob for tube-vs-plate confusion, per skimage's own docstring."""
    filtered_max = None
    for eigvals in eigvals_by_sigma:
        lambda1 = eigvals[0]
        lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
        r_a = lambda2 / lambda3
        r_b = np.abs(lambda1) / np.sqrt(lambda2 * lambda3)
        s = np.sqrt((eigvals ** 2).sum(0))
        vals = 1.0 - np.exp(-(r_a ** 2) / (2 * alpha ** 2))
        vals *= np.exp(-(r_b ** 2) / (2 * beta ** 2))
        vals *= 1.0 - np.exp(-(s ** 2) / (2 * gamma ** 2))
        filtered_max = vals if filtered_max is None else np.maximum(filtered_max, vals)
    return filtered_max
