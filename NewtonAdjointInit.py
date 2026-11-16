#Import libraries-
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image

from RabbitPathFinder import find_all_the_paths
from ComposedTransformReg import ComposedReg

# First pass at the normalized-adjoint initialization from the colleague's
# "Inverse-Transformed Image as an Inverse Problem" doc (Eq. 29-33), as a
# trilinear-interpolation replacement for PaintHnEFeatureToInVivo.py's hard
# nearest-neighbor voxel-ID + cell-count-weighted-average pipeline.
#
# Scope for now: a single registered H&E slide, grayscale intensity as the
# placeholder feature, Q used purely as a coverage mask (1 where we have real
# H&E data, 0 elsewhere) rather than cell-count weighting. Multi-slide
# assembly and real features/labels come later.


def load_moving_vox(RabbitID, Block, RabbitFolder, MovingStart='InVivo', EndFixed='BlockFace',
                     return_target_affine=False):
    """h: for every BlockFace(target)-grid voxel, the continuous MR(source)-voxel-index
    coordinate it pulls from. Reuses ComposedReg's own save_moving_vox_path output
    instead of re-deriving the point-pull chain here."""
    with tempfile.TemporaryDirectory() as tmp:
        save_path = str(Path(tmp) / 'moving_vox.nii.gz')
        ComposedReg(RabbitID, Block, RabbitFolder, MovingStart, EndFixed,
                    interpolation='linear', save=False, save_moving_vox_path=save_path)
        base = save_path[:-len('.nii.gz')]
        i_nib = nib.load(f"{base}_i.nii.gz")
        i = i_nib.get_fdata()
        j = nib.load(f"{base}_j.nii.gz").get_fdata()
        k = nib.load(f"{base}_k.nii.gz").get_fdata()
    h = np.stack([i, j, k], axis=-1)  # (X,Y,Z,3)
    if return_target_affine:
        return h, i_nib.affine
    return h


def load_source_geometry(RabbitID, Block, RabbitFolder, MovingStart='InVivo'):
    """Shape and affine of the MR (source) grid that `a` lives on — same
    canonicalization ComposedReg applies internally, so indices line up with
    moving_vox. The affine is what makes a saved `a` land in the right place
    physically — do not save `a` with anything else (e.g. identity)."""
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
    moving_nib = nib.as_closest_canonical(nib.load(paths['Moving_FilePath']))
    return moving_nib.shape[:3], moving_nib.affine


def load_native_geometry(RabbitID, Block, RabbitFolder, MovingStart='InVivo'):
    """Shape and affine of the same MR (source) file as load_source_geometry,
    but in its raw on-disk orientation (no canonicalization) -- the
    convention Day3End_Registered/ExtractMRFeatures.py already use. Used to
    reorient a finished `a` (computed in load_source_geometry's canonical
    space) back to this convention before saving, so it lands voxel-for-voxel
    aligned with the rest of the MR pipeline with no reorientation needed on
    the consumer side."""
    paths = find_all_the_paths(RabbitID, Block, RabbitFolder, MovingStart)
    moving_nib = nib.load(paths['Moving_FilePath'])
    return moving_nib.shape[:3], moving_nib.affine


def find_z_index(cropped_images_dir, img_number):
    """Position of IMG_{num:04d}_*.tiff in the sorted CroppedImages listing —
    this is exactly the ordering BlockFaceToNifti stacks into the Z axis."""
    files = sorted(f for f in Path(cropped_images_dir).glob('*.tiff') if not f.name.startswith('._'))
    prefix = f"IMG_{img_number:04d}_"
    matches = [idx for idx, f in enumerate(files) if f.name.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one CroppedImages match for {prefix}, got {matches}"
    return matches[0]


def load_hne_slide_downsampled(png_path):
    """Grayscale intensity, decimated by 4 in X/Y via plain striding — same
    convention as BlockFaceToNifti's vol[::4,::4,:] (no averaging), so this
    lands exactly on the existing 4x-downsampled BlockFace grid with no offset."""
    grey = np.array(Image.open(png_path).convert('L')).astype(np.float32)  # (H,W)
    grey = grey.T  # -> (W,H) = (X,Y), matching BlockFaceToNifti's axis convention
    return grey[::4, ::4]


def discover_img_numbers(registered_hne_dir):
    """All available registered slide numbers (e.g. 17, 22, 27, ...) parsed
    from IMG{num}reg.png filenames in the directory, sorted ascending."""
    import re
    numbers = []
    for f in Path(registered_hne_dir).glob('IMG*reg.png'):
        if f.name.startswith('._'):
            continue
        m = re.match(r'IMG(\d+)reg\.png$', f.name)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)


def build_b_and_Q(registered_hne_dir, cropped_images_dir, target_shape, img_numbers, target_affine):
    """b: H&E grayscale intensity placed at its true Z-slice, zero elsewhere.
    Q: 1 at voxels with real H&E data, 0 elsewhere — doubles as the coverage
    mask for sparse Z-coverage, same Q the paper's weighted adjoint expects.

    Flips X/Y when target_affine is positive-diagonal -- same correction
    TilestoVoxelsV2.py already applies for this grid: BlockFaceToNifti's true
    ground-truth affine (what the real BlockFace images are saved with) is
    negative-diagonal with no array flip, so a positive-diagonal target grid
    needs this flip to land in the same real-world orientation."""
    X, Y, Z = target_shape
    b = np.zeros((X, Y, Z), dtype=np.float64)
    Q = np.zeros((X, Y, Z), dtype=np.float64)
    for img_number in img_numbers:
        z = find_z_index(cropped_images_dir, img_number)
        png_path = Path(registered_hne_dir) / f"IMG{img_number:04d}reg.png"
        slide = load_hne_slide_downsampled(png_path)
        sx, sy = slide.shape
        b[:sx, :sy, z] = slide
        Q[:sx, :sy, z] = 1.0
    if target_affine[0, 0] > 0 and target_affine[1, 1] > 0:
        b = b[::-1, ::-1, :]
        Q = Q[::-1, ::-1, :]
    return b, Q


def _trilinear_stencil(q, source_shape):
    """8 integer corner coords + weights per target voxel j, following the
    doc's Eq. 12-13 notation: q is q_j (continuous source index, already
    h(y_j) converted to index space), k_j = floor(q_j), theta_j = q_j - k_j.
    Zero-extension boundary policy: out-of-bounds corners get valid=False and
    contribute nothing — same policy must be used for both the gather and
    the scatter, or adjointness breaks (per the doc's section 3.2)."""
    k_x = np.floor(q[..., 0]).astype(np.int64)
    k_y = np.floor(q[..., 1]).astype(np.int64)
    k_z = np.floor(q[..., 2]).astype(np.int64)
    theta_x = q[..., 0] - k_x
    theta_y = q[..., 1] - k_y
    theta_z = q[..., 2] - k_z

    corners = []
    for e_x in (0, 1):
        for e_y in (0, 1):
            for e_z in (0, 1):
                corner_x = k_x + e_x
                corner_y = k_y + e_y
                corner_z = k_z + e_z
                w_e = ((theta_x if e_x else 1 - theta_x) *
                       (theta_y if e_y else 1 - theta_y) *
                       (theta_z if e_z else 1 - theta_z))
                valid = ((corner_x >= 0) & (corner_x < source_shape[0]) &
                         (corner_y >= 0) & (corner_y < source_shape[1]) &
                         (corner_z >= 0) & (corner_z < source_shape[2]))
                corners.append((corner_x, corner_y, corner_z, w_e, valid))
    return corners


def apply_A(a, corners):
    """Forward trilinear gather (Eq. 14): a lives on the MR (source) grid,
    corners is a precomputed _trilinear_stencil(q, a.shape) — pass the same
    corners into apply_A_adjoint rather than recomputing per call, since q is
    fixed for the whole run (per the doc's section 6.4 caching remark)."""
    target_shape = corners[0][0].shape
    out = np.zeros(target_shape, dtype=np.float64)
    for corner_x, corner_y, corner_z, w_e, valid in corners:
        out[valid] += w_e[valid] * a[corner_x[valid], corner_y[valid], corner_z[valid]]
    return out


def apply_A_adjoint(r, corners, source_shape):
    """Adjoint scatter: same precomputed corners/w_{j,e} weights as the
    forward gather, scattered from target residual r back onto the source
    grid. Correctly sums (not overwrites) contributions from multiple target
    voxels landing on the same source voxel, via np.bincount on each
    corner's flattened source index -- np.add.at does the same sum but is a
    well-known numpy performance trap (falls back to an unvectorized
    per-element loop precisely because it handles repeated indices), while
    bincount has a dedicated fast path for exactly this "sum weighted values
    into integer bins" pattern. Flattening/reshaping is purely a change of
    index representation (3D coord -> 1D bin -> back to 3D) to fit
    bincount's 1D-only interface; the values summed and where they land are
    identical either way."""
    nx, ny, nz = source_shape
    out_flat = np.zeros(nx * ny * nz, dtype=np.float64)
    for corner_x, corner_y, corner_z, w_e, valid in corners:
        flat_idx = (corner_x[valid] * ny + corner_y[valid]) * nz + corner_z[valid]
        out_flat += np.bincount(flat_idx, weights=w_e[valid] * r[valid], minlength=out_flat.size)
    return out_flat.reshape(source_shape)


def normalized_adjoint_init(b, Q, corners, source_shape, a0=None, tau=1e-6):
    """Eq. 32-33: d = A*_h(Q*1), n = A*_h(Q*b), a0-fallback where coverage <= tau."""
    if a0 is None:
        a0 = np.zeros(source_shape, dtype=np.float64)
    coverage = apply_A_adjoint(Q, corners, source_shape)
    numerator = apply_A_adjoint(Q * b, corners, source_shape)
    a = a0.copy()
    covered = coverage > tau
    a[covered] = numerator[covered] / coverage[covered]
    return a, coverage


def check_adjoint(corners, source_shape, target_shape, rng=None, n_trials=1):
    """Eq. 34 adjoint test — random a/r, verify <A a, r> == <a, A* r> to
    near machine precision. The paper flags this as the single most direct
    check of the adjoint implementation; run manually before trusting
    normalized_adjoint_init's output."""
    rng = rng or np.random.default_rng(0)
    for _ in range(n_trials):
        a = rng.standard_normal(source_shape)
        r = rng.standard_normal(target_shape)
        Aa = apply_A(a, corners)
        Astar_r = apply_A_adjoint(r, corners, source_shape)
        lhs = np.sum(Aa * r)
        rhs = np.sum(a * Astar_r)
        denom = (np.linalg.norm(Aa) * np.linalg.norm(r)
                 + np.linalg.norm(a) * np.linalg.norm(Astar_r) + 1e-12)
        rel_err = abs(lhs - rhs) / denom
        print(f"adjoint test: <Aa,r>={lhs:.6e}  <a,A*r>={rhs:.6e}  rel_err={rel_err:.3e}")


def apply_hessian(v, corners, Q, source_shape, lam):
    """Hv = A*_h(Q * (A_h v)) + lambda*v — Eq. 42, the matrix-free Hessian
    action. Never assembled as a matrix; every CG iteration is one forward
    gather + one adjoint scatter (Eq. 46)."""
    return apply_A_adjoint(Q * apply_A(v, corners), corners, source_shape) + lam * v


def objective(a, b, Q, corners, a0, lam):
    """Phi(a) = 0.5||Q^{1/2}(A_h a - b)||^2 + 0.5*lambda*||a-a0||^2 — Eq. 9."""
    r = apply_A(a, corners) - b
    return 0.5 * np.sum(Q * r * r) + 0.5 * lam * np.sum((a - a0) ** 2)


def conjugate_gradient(g, corners, Q, source_shape, lam, eta, max_iters=200):
    """Approximately solve H p = -g (Eqs. 47-52), stopping once the residual
    has shrunk to eta * ||g|| — the inexact-Newton condition, Eq. 53."""
    p = np.zeros_like(g)
    s = -g.copy()
    d = s.copy()
    g_norm = np.linalg.norm(g)
    s_dot_s = np.dot(s.ravel(), s.ravel())
    n_iters = 0
    for _ in range(max_iters):
        if np.sqrt(s_dot_s) <= eta * g_norm:
            break
        q = apply_hessian(d, corners, Q, source_shape, lam)
        dq = np.dot(d.ravel(), q.ravel())
        if dq <= 0:
            # H is guaranteed SPD for lambda > 0 (Eq. 43); non-positive
            # curvature here means numerical trouble, not a valid CG step.
            break
        alpha = s_dot_s / dq
        p = p + alpha * d
        s_new = s - alpha * q
        s_new_dot_s_new = np.dot(s_new.ravel(), s_new.ravel())
        beta = s_new_dot_s_new / s_dot_s
        d = s_new + beta * d
        s, s_dot_s = s_new, s_new_dot_s_new
        n_iters += 1
    print(f"    CG: stopped after {n_iters} iterations (eta={eta}, max_iters={max_iters})")
    return p


def newton_cg_solve(b, Q, corners, source_shape, a0, lam=1.0, tol=1e-4,
                     eta=0.25, max_outer_iters=20, c1=1e-4, iterate_callback=None):
    """Solves Eq. 9 for a_lambda via matrix-free Newton-CG (section 6.4),
    starting from a0 as both the initial iterate and the regularizer's prior.

    lam/tol/eta are placeholders, not tuned values — how to concretely set
    lambda (and eventually Q as a real confidence weight, and a0 as a richer
    prior) for the real MR/H&E case is still open; revisit once this is
    validated end-to-end on the placeholder intensity feature.

    For this exactly-quadratic objective a single outer iteration with a
    well-converged CG solve reaches the minimizer (section 6.3) — the outer
    loop and backtracking are kept general so this extends unchanged to the
    non-quadratic robust data term in section 7 later.

    iterate_callback(k, a), if given, is called with the pre-update iterate at
    the start of every outer pass (k=0 is a0 itself) -- for inspecting
    per-iteration convergence, e.g. saving each a_k to disk."""
    a = a0.copy()
    b_adj_norm = np.linalg.norm(apply_A_adjoint(Q * b, corners, source_shape))
    for k in range(max_outer_iters):
        if iterate_callback is not None:
            iterate_callback(k, a)
        r = apply_A(a, corners) - b
        g = apply_A_adjoint(Q * r, corners, source_shape) + lam * (a - a0)
        g_norm = np.linalg.norm(g)
        print(f"  outer iter {k}: ||g||={g_norm:.6e}  Phi(a)={objective(a, b, Q, corners, a0, lam):.6e}")
        if g_norm <= tol * max(1, b_adj_norm):
            break

        p = conjugate_gradient(g, corners, Q, source_shape, lam, eta)

        t = 1.0
        phi_a = objective(a, b, Q, corners, a0, lam)
        gp = np.dot(g.ravel(), p.ravel())
        while objective(a + t * p, b, Q, corners, a0, lam) > phi_a + c1 * t * gp and t > 1e-8:
            t /= 2
        a = a + t * p
    return a


def poisson_i_divergence(a, b, Q, corners, tau=1e-9):
    """Csiszar's I-divergence I(b || A_h a), restricted to Q-valid target
    voxels -- the Poisson log-likelihood of Eq. 45 up to an additive
    constant. Should DECREASE every Richardson-Lucy iteration (theory
    guarantee); used the same way check_adjoint is used for Newton-CG, as a
    cheap sanity check that the implementation actually matches the math."""
    Aa = apply_A(a, corners)
    mask = Q > 0
    b_valid = b[mask]
    Aa_valid = np.maximum(Aa[mask], tau)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(b_valid > 0, b_valid * np.log(np.maximum(b_valid, tau) / Aa_valid), 0.0)
    return float(np.sum(term - b_valid + Aa_valid))


def richardson_lucy_solve(b, Q, corners, source_shape, tau=1e-6, max_iters=200,
                           range_tolerance=0.01, enforce_range_stop=True,
                           clamp_out_of_range=False, iterate_callback=None):
    """Positive image reconstruction via Richardson-Lucy EM (Eq. 45-53),
    beta=0 (no image prior -- see MAP-EM, section 7.6.4, for a later
    extension that would add one). Unlike newton_cg_solve, this has no a0
    argument: RL's multiplicative update can never move a voxel away from
    exactly 0 (0 * anything = 0), so every voxel needs a strictly-positive
    starting value, not a zero fallback. Warm-started via
    normalized_adjoint_init with its own fallback set to the observed data's
    minimum (instead of 0) -- covered voxels get a real data-driven estimate,
    uncovered voxels get that same positive floor, and since every real b
    value is >= that floor, the weighted average at covered voxels is too
    (convex combination), so strict positivity holds everywhere by
    construction.

    Stopping: RL has no natural residual/gradient-based stopping criterion of
    its own -- its "regularization" IS the iteration count (semi-convergence,
    per the doc's section 10 comparison table: "noise grows with iteration").
    Rather than pick an arbitrary fixed iteration count, we stop the first
    time the reconstruction's own value range drifts outside the observed
    data's range (with `range_tolerance` slack on the span) and return the
    PRIOR iterate -- so the result never actually leaves that range. This
    check is image-wide (a single voxel leaving range stops the whole
    volume) for the STOP decision -- simpler, kept for that purpose. Set
    enforce_range_stop=False to keep iterating past that point instead --
    e.g. for sweeping iteration count -- while still logging (once) the
    iteration where the stop would have fired, for reference.

    clamp_out_of_range=True switches the CONSEQUENCE from image-wide to
    per-voxel: instead of stopping (or silently continuing unclamped), every
    voxel that left [allowed_lo, allowed_hi] this iteration is clipped back
    to the boundary it crossed, elementwise, while every other voxel is left
    untouched. Voxels pinned this way can still recover on a later iteration
    if their local update factor drops back under 1 -- nothing is permanently
    excluded from further updates. Takes precedence over enforce_range_stop
    when both are set (clamps instead of breaking).

    iterate_callback(k, a), if given, is called with the pre-update iterate
    at the start of every iteration (k=0 is the initial image itself)."""
    valid_b = b[Q > 0]
    lo, hi = valid_b.min(), valid_b.max()
    span = hi - lo
    allowed_lo = lo - range_tolerance * span
    allowed_hi = hi + range_tolerance * span
    print(f"  RL: data range=[{lo:.3f}, {hi:.3f}], allowed=[{allowed_lo:.3f}, {allowed_hi:.3f}]")

    a0_fallback = np.full(source_shape, lo, dtype=np.float64)
    a, coverage = normalized_adjoint_init(b, Q, corners, source_shape, a0=a0_fallback, tau=tau)
    covered = coverage > tau

    prev_div = None
    would_have_stopped_at = None
    for k in range(max_iters):
        if iterate_callback is not None:
            iterate_callback(k, a)

        Aa = apply_A(a, corners)
        safe = (Q > 0) & (Aa > tau)
        ratio = np.zeros_like(Aa)
        ratio[safe] = b[safe] / Aa[safe]
        scattered = apply_A_adjoint(ratio, corners, source_shape)  # A*Q(b (/) Aa); Q already folded in via `safe`

        a_next = a.copy()
        a_next[covered] = a[covered] * (scattered[covered] / coverage[covered])

        div = poisson_i_divergence(a_next, b, Q, corners)
        flag = "" if prev_div is None or div <= prev_div + 1e-6 else "  <-- INCREASED, check implementation"
        print(f"  RL iter {k}: I-divergence={div:.6e}{flag}")
        prev_div = div

        out_of_range = a_next.min() < allowed_lo or a_next.max() > allowed_hi
        if out_of_range and would_have_stopped_at is None:
            would_have_stopped_at = k
            print(f"  RL: iter {k} left allowed range "
                  f"(min={a_next.min():.3f}, max={a_next.max():.3f})"
                  f"{' -- clamping out-of-range voxels' if clamp_out_of_range else ' -- stopping' if enforce_range_stop else ' -- would stop here, continuing anyway'}")
        if out_of_range and clamp_out_of_range:
            a_next = np.clip(a_next, allowed_lo, allowed_hi)
        elif out_of_range and enforce_range_stop:
            break
        a = a_next
    return a


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID = "R24-240"
    Block = 3
    USE_CHECKERBOARD = True  # same toggle/folder convention as RichardsonLucyTest.py, so outputs land together

    if USE_CHECKERBOARD:
        registered_hne_dir = str(Path(__file__).parent / 'TestData' / f"{RabbitID}_Block{Block:02d}_CheckerboardSynthetic")
    else:
        registered_hne_dir = os.path.join(RabbitFolder, RabbitID, 'HnE', f"Block{Block:02d}", 'Registered')
    cropped_images_dir = find_all_the_paths(RabbitID, Block, RabbitFolder, 'ExVivoBlock')['Fixed_Folder']
    cropped_images_dir = os.path.join(cropped_images_dir, 'CroppedImages')

    out_dir = str(Path(__file__).parent / 'TestData' /
                  (f"{RabbitID}_Block{Block:02d}_CheckerboardReconComparison" if USE_CHECKERBOARD
                   else f"{RabbitID}_Block{Block:02d}_ReconComparison"))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading h (moving_vox)...")
    #Run composed reg to save moving voxel map then reload here->
    h, target_affine = load_moving_vox(RabbitID, Block, RabbitFolder, return_target_affine=True)
    target_shape = h.shape[:3]
    source_shape, source_affine = load_source_geometry(RabbitID, Block, RabbitFolder)
    print(f"target (BlockFace) shape: {target_shape}, source (MR) shape: {source_shape}")

    img_numbers = discover_img_numbers(registered_hne_dir)
    print(f"Building b, Q from {len(img_numbers)} registered slides: {img_numbers}...")
    b, Q = build_b_and_Q(registered_hne_dir, cropped_images_dir, target_shape, img_numbers, target_affine)

    b_out_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_b_test.nii.gz")
    nib.save(nib.Nifti1Image(b.astype(np.float32), target_affine), b_out_path)
    print(f"Saved b (H&E projected onto BlockFace target grid) to {b_out_path} for manual inspection.")

    print("Computing trilinear stencil (once — h is fixed, reused below)...")
    corners = _trilinear_stencil(h, source_shape)

    print("Running adjoint test...")
    check_adjoint(corners, source_shape, target_shape)

    print("Computing normalized-adjoint initialization...")
    a0, coverage = normalized_adjoint_init(b, Q, corners, source_shape)
    print(f"a0 stats: min={a0.min():.3f} max={a0.max():.3f} mean={a0.mean():.3f} "
          f"covered voxels={int((coverage > 1e-6).sum()):,}/{a0.size:,}")

    out_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_a0_test.nii.gz")
    nib.save(nib.Nifti1Image(a0.astype(np.float32), source_affine), out_path)
    print(f"Saved initial reconstruction to {out_path} for manual inspection.")

    print("Solving Newton-CG (lam/tol/eta are placeholders — see newton_cg_solve docstring)...")
    a = newton_cg_solve(b, Q, corners, source_shape, a0, lam=100.0, tol=1e-4, eta=0.25)
    print(f"a stats: min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f}")

    a_out_path = os.path.join(out_dir, f"{RabbitID}_Block{Block:02d}_NewtonCGRecon.nii.gz")
    nib.save(nib.Nifti1Image(a.astype(np.float32), source_affine), a_out_path)
    print(f"Saved Newton-CG reconstruction to {a_out_path} for manual inspection.")
