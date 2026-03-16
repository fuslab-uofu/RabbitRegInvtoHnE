import sys
import nibabel as nib
import numpy as np
from scipy.ndimage import label, center_of_mass, binary_dilation
from scipy.spatial import Delaunay
from skimage.filters import threshold_otsu
from PyQt5.QtWidgets import QApplication
from Viewer import VolumeViewer


def segment_ablation(arr, pct_threshold=99.99):
    """Binary mask of the largest high-intensity connected component.

    Uses a high-percentile threshold to isolate hyperintense tissue (ablation),
    then returns only the largest connected component to exclude scattered bright spots.
    """
    thresh = np.percentile(arr, pct_threshold)
    binary = arr >= thresh
    labeled, n = label(binary)
    if n == 0:
        return np.zeros_like(arr, dtype=bool)
    sizes = np.array([np.sum(labeled == i) for i in range(1, n + 1)])
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def segment_from_seed(arr, seed, roi_halfwidth=40, intensity_fraction=0.75):
    """Segment the ablation using a user-provided seed point.

    Thresholds a local ROI at a fraction of the seed voxel's intensity, so the
    threshold auto-calibrates to wherever the user clicks rather than relying on
    global statistics. Returns the connected component at or nearest to the seed.

    If the seed lands in a dark region (e.g. inside the ablation rather than on
    the bright shell), it snaps to the nearest bright voxel in the ROI first.
    """
    seed = tuple(int(round(s)) for s in seed)

    roi_slices = tuple(
        slice(max(0, seed[i] - roi_halfwidth), min(arr.shape[i], seed[i] + roi_halfwidth))
        for i in range(3)
    )
    roi = arr[roi_slices]
    seed_in_roi = tuple(seed[i] - roi_slices[i].start for i in range(3))

    seed_val = roi[seed_in_roi]

    # If seed is in a dark region, snap to the nearest tissue-level voxel
    if seed_val < threshold_otsu(roi):
        bright_for_snap = roi >= threshold_otsu(roi)
        bright_coords = np.array(np.where(bright_for_snap)).T
        if len(bright_coords) == 0:
            return np.zeros_like(arr, dtype=bool)
        dists = np.linalg.norm(bright_coords - np.array(seed_in_roi), axis=1)
        nearest = bright_coords[np.argmin(dists)]
        seed_val = roi[tuple(nearest)]
        seed_in_roi = tuple(nearest)

    # Find voxels with similar brightness to the seed
    bright = roi >= seed_val * intensity_fraction

    labeled, n = label(bright)
    if n == 0:
        return np.zeros_like(arr, dtype=bool)

    seed_label = labeled[seed_in_roi]

    full_mask = np.zeros_like(arr, dtype=bool)
    full_mask[roi_slices] = labeled == seed_label
    return full_mask


def segment_from_seed_gradient(arr, seed, roi_halfwidth=60, shell_search_radius=20):
    """Segment the ablation zone using gradient magnitude from a seed inside the ablation.

    Computes gradient magnitude in a local ROI (np.gradient uses centered differences).
    The ablation shell appears as a ring of high gradient magnitude with two faces:
    an inner wall (dark→bright transition) and an outer wall (bright→dark).

    Steps:
    1. Find the edge component nearest to the seed (inner wall).
    2. Dilate that component by shell_search_radius to span the shell thickness
       and capture the outer wall too.
    3. Collect all edge voxels within the dilated region → full shell boundary.
    4. Return the convex hull of those boundary points as the ablation zone.

    shell_search_radius: dilation in voxels to bridge inner→outer wall. At 0.25 mm/voxel,
    20 voxels = 5 mm, which covers a shell up to ~4 mm thick.
    """
    seed = tuple(int(round(s)) for s in seed)

    roi_slices = tuple(
        slice(max(0, seed[i] - roi_halfwidth), min(arr.shape[i], seed[i] + roi_halfwidth))
        for i in range(3)
    )
    roi = arr[roi_slices]
    seed_in_roi = tuple(seed[i] - roi_slices[i].start for i in range(3))

    # Gradient magnitude: high at the ablation shell boundary
    grads = np.gradient(roi)
    grad_mag = np.sqrt(sum(g**2 for g in grads))

    # Otsu threshold separates edge voxels (high gradient) from flat regions
    thresh = threshold_otsu(grad_mag)
    edge_mask = grad_mag >= thresh

    if not np.any(edge_mask):
        return np.zeros_like(arr, dtype=bool)

    # Find the edge component nearest to the seed (inner wall of ablation shell)
    labeled, _ = label(edge_mask)
    edge_coords = np.column_stack(np.where(edge_mask))
    dists = np.linalg.norm(edge_coords - np.array(seed_in_roi), axis=1)
    nearest_label = labeled[tuple(edge_coords[np.argmin(dists)])]
    inner_wall = labeled == nearest_label

    # Dilate the inner wall to span the shell thickness and capture the outer wall
    dilated = binary_dilation(inner_wall, iterations=shell_search_radius)

    # Collect edge voxels within the dilated region — these form the full shell boundary
    shell_coords = edge_coords[dilated[tuple(edge_coords.T)]].astype(float)

    if len(shell_coords) < 4:
        return np.zeros_like(arr, dtype=bool)

    # Convex hull of the shell defines the full ablation zone (interior + shell)
    try:
        hull = Delaunay(shell_coords)
    except Exception:
        return np.zeros_like(arr, dtype=bool)

    lo = np.maximum(0, shell_coords.min(axis=0).astype(int))
    hi = np.minimum(np.array(roi.shape) - 1, shell_coords.max(axis=0).astype(int) + 1)
    x = np.arange(lo[0], hi[0] + 1)
    y = np.arange(lo[1], hi[1] + 1)
    z = np.arange(lo[2], hi[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(float)

    in_hull = hull.find_simplex(voxels) >= 0
    coords_in = voxels[in_hull].astype(int)

    roi_mask = np.zeros(roi.shape, dtype=bool)
    roi_mask[coords_in[:, 0], coords_in[:, 1], coords_in[:, 2]] = True

    full_mask = np.zeros_like(arr, dtype=bool)
    full_mask[roi_slices] = roi_mask
    return full_mask


def compute_dsc(mask1, mask2):
    """Dice Similarity Coefficient between two binary masks. Returns 0.0-1.0."""
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_centroid_distance_mm(mask1, mask2, zooms):
    """Euclidean distance in mm between centroids of two binary masks."""
    c1 = np.array(center_of_mass(mask1)) * np.array(zooms)
    c2 = np.array(center_of_mass(mask2)) * np.array(zooms)
    return float(np.linalg.norm(c1 - c2))


def assess_stage(registered_path, fixed_path, pct_threshold=90):
    """Assess registration quality for one stage using the ablation region.

    Segments the hyperintense ablation in both the registered moving image and
    the fixed image, then computes DSC and centroid distance.

    The fixed image is canonicalized on load to match the convention used when
    generating the registered output (LandMarker operates in canonical space).
    """
    reg_img = nib.load(registered_path)
    reg_arr = reg_img.get_fdata()

    fixed_img = nib.load(fixed_path)
    fixed_can = nib.as_closest_canonical(fixed_img)
    fixed_arr = fixed_can.get_fdata()

    zooms = fixed_can.header.get_zooms()
    vox_vol_mm3 = float(np.prod(zooms))

    reg_mask = segment_ablation(reg_arr, pct_threshold)
    fixed_mask = segment_ablation(fixed_arr, pct_threshold)

    dsc = compute_dsc(reg_mask, fixed_mask)
    centroid_dist_mm = compute_centroid_distance_mm(reg_mask, fixed_mask, zooms)

    return {
        'dsc': dsc,
        'centroid_distance_mm': centroid_dist_mm,
        'ablation_volume_registered_mm3': float(np.sum(reg_mask)) * vox_vol_mm3,
        'ablation_volume_fixed_mm3': float(np.sum(fixed_mask)) * vox_vol_mm3,
        'reg_arr': reg_arr,
        'fixed_arr': fixed_arr,
        'reg_mask': reg_mask,
        'fixed_mask': fixed_mask,
    }


def norm255(arr):
    m = np.max(arr)
    return arr / m * 255 if m > 0 else arr


if __name__ == '__main__':
    registered_path = '/Users/jbonaventura/Downloads/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToExVivo_0304-1231.nii.gz'
    fixed_path = '/Users/jbonaventura/Downloads/RabbitData/R23-055/ExVivo_MR/ExVivo.nii'

    reg_img = nib.load(registered_path)
    reg_arr = reg_img.get_fdata()

    fixed_img = nib.load(fixed_path)
    fixed_can = nib.as_closest_canonical(fixed_img)
    fixed_arr = fixed_can.get_fdata()
    zooms = fixed_can.header.get_zooms()

    reg_mask = np.zeros_like(reg_arr, dtype=bool)
    fixed_mask = np.zeros_like(fixed_arr, dtype=bool)

    def on_reg_click(x, y, z):
        print(f'Registered image seed: ({x}, {y}, {z}) — segmenting...')
        reg_mask[:] = segment_from_seed_gradient(reg_arr, (x, y, z))
        print(f'  Ablation zone: {np.sum(reg_mask) * float(np.prod(zooms)):.1f} mm3')
        viewer_reg.data2 = reg_mask * 255
        viewer_reg.update_plot()

    def on_fixed_click(x, y, z):
        print(f'Fixed image seed: ({x}, {y}, {z}) — segmenting...')
        fixed_mask[:] = segment_from_seed_gradient(fixed_arr, (x, y, z))
        vox_vol = float(np.prod(zooms))
        dsc = compute_dsc(reg_mask, fixed_mask)
        dist = compute_centroid_distance_mm(reg_mask, fixed_mask, zooms)
        print('Registration assessment:')
        print(f'  DSC:                          {dsc:.4f}')
        print(f'  Centroid distance:            {dist:.2f} mm')
        print(f'  Ablation volume (registered): {np.sum(reg_mask) * vox_vol:.1f} mm3')
        print(f'  Ablation volume (fixed):      {np.sum(fixed_mask) * vox_vol:.1f} mm3')
        viewer_fixed.data2 = fixed_mask * 255
        viewer_fixed.update_plot()

    app = QApplication(sys.argv)
    viewer_reg = VolumeViewer(norm255(reg_arr), np.zeros_like(reg_arr),
                              'Registered InVivo — click inside ablation to seed',
                              seed_callback=on_reg_click)
    viewer_reg.show()
    viewer_fixed = VolumeViewer(norm255(fixed_arr), np.zeros_like(fixed_arr),
                                'Fixed ExVivo — click inside ablation to seed',
                                seed_callback=on_fixed_click)
    viewer_fixed.move(820, 100)
    viewer_fixed.show()
    sys.exit(app.exec_())
