import os
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from skimage.graph import route_through_array
from PyQt5.QtWidgets import QApplication, QFileDialog
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


def _fill_cube(mask, centre, radius):
    """Set a cubic region of voxels in mask to True around centre (in-place)."""
    cx, cy, cz = (int(round(c)) for c in centre)
    r = radius
    x0, x1 = max(0, cx - r), min(mask.shape[0], cx + r + 1)
    y0, y1 = max(0, cy - r), min(mask.shape[1], cy + r + 1)
    z0, z1 = max(0, cz - r), min(mask.shape[2], cz + r + 1)
    mask[x0:x1, y0:y1, z0:z1] = True


def _erase_cube(mask, centre, radius, slice_axis=None, z_radius=1):
    """Zero out a region of voxels in mask around centre (in-place).

    If slice_axis is given, radius applies to the two in-plane axes and z_radius
    controls depth along slice_axis.  If slice_axis is None, radius applies to
    all axes (symmetric cube removal).
    """
    centre = tuple(int(round(c)) for c in centre)
    slices = []
    for ax in range(mask.ndim):
        r = (z_radius if ax == slice_axis else radius) if slice_axis is not None else radius
        slices.append(slice(max(0, centre[ax] - r), min(mask.shape[ax], centre[ax] + r + 1)))
    mask[tuple(slices)] = False


def segment_from_seeds(arr, seeds, cuts=None, hi_bound=None):
    """Segment ablation by intensity range defined by multiple seed points.

    Thresholds to [min_seed, hi_bound] where hi_bound defaults to max seed
    intensity. Optionally erases cubes around cut points (each cut is a
    (x, y, z, radius) tuple) to sever connections, then keeps only connected
    components containing a seed point.
    """
    if not seeds:
        return np.zeros_like(arr, dtype=bool)
    seeds = [tuple(int(round(c)) for c in s) for s in seeds]
    intensities = [arr[s] for s in seeds]
    lo = min(intensities)
    hi = hi_bound if hi_bound is not None else max(intensities)
    threshold_mask = (arr >= lo) & (arr <= hi)
    if cuts:
        for cut in cuts:
            cx, cy, cz, r = cut[:4]
            slice_axis = cut[4] if len(cut) > 4 else None
            _erase_cube(threshold_mask, (cx, cy, cz), r, slice_axis)
    labeled, _ = label(threshold_mask)
    seed_labels = {labeled[s] for s in seeds if labeled[s] > 0}
    result = np.zeros_like(arr, dtype=bool)
    for lbl in seed_labels:
        result |= (labeled == lbl)
    return result

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

    return {
        'reg_arr': reg_arr,
        'fixed_arr': fixed_arr,
        'reg_mask': reg_mask,
        'fixed_mask': fixed_mask,
    }


def flatten_field(arr, shrink_factor=4):
    """Apply N4 bias field correction to remove slow-varying intensity non-uniformity.

    Computes N4 on a downsampled image for speed, then projects the bias field
    back to full resolution before dividing out.
    """
    sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
    # Clamp per-dimension so no axis shrinks below 2 voxels
    safe_factors = [min(shrink_factor, max(1, d // 2)) for d in sitk_img.GetSize()]
    shrunken = sitk.Shrink(sitk_img, safe_factors)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(shrunken)
    log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
    corrected = sitk_img / sitk.Exp(log_bias)
    return sitk.GetArrayFromImage(corrected)


def bridge_path(arr, start, end, radius=1):
    """Find the minimum cost path between start and end through bright voxels.

    Cost is inverse intensity so bright regions are cheap to traverse.
    Returns a boolean mask with the path dilated by radius set to True.
    """
    cost = 1.0 / (arr.astype(float) + 1e-6)
    path, _ = route_through_array(cost, start, end, fully_connected=True)
    bridge = np.zeros_like(arr, dtype=bool)
    for pt in path:
        _fill_cube(bridge, pt, radius)
    return bridge


def make_seg_nifti(mask, affine, header=None):
    """Create a uint8 NIfTI image from a boolean segmentation mask."""
    return nib.Nifti1Image(mask.astype(np.uint8), affine, header)


def norm255(arr):
    m = np.max(arr)
    return arr / m * 255 if m > 0 else arr


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Mutable state — use a dict so closures can reassign without nonlocal
    state = {'arr': None, 'nib_img': None, 'mask': None, 'dir': ''}
    seeds = []
    cuts = []
    bridges = []      # list of boolean mask arrays, one per bridge
    action_log = []

    def _load_nifti(path):
        """Load, canonicalize, and N4-correct a NIfTI. Updates state in place."""
        print(f'Loading {path}')
        img = nib.as_closest_canonical(nib.load(path))
        print('Applying N4 bias field correction...')
        arr = flatten_field(img.get_fdata())
        print('Done.')
        state['arr'] = arr
        state['nib_img'] = img
        state['mask'] = np.zeros_like(arr, dtype=bool)
        state['dir'] = os.path.dirname(os.path.abspath(path))

    def _reset_session():
        """Clear all seeds, cuts, bridges, and mask for a fresh segmentation."""
        seeds.clear()
        cuts.clear()
        bridges.clear()
        action_log.clear()
        state['mask'][:] = False

    def _resegment():
        hi = float(np.max(state['arr'])) if viewer.expand_to_max else None
        seg = segment_from_seeds(state['arr'], seeds, cuts=cuts, hi_bound=hi)
        for b in bridges:
            seg |= b
        state['mask'][:] = seg
        viewer.data2 = state['mask'].astype(float) * 255
        viewer.update_plot()

    def on_click(x, y, z):
        seeds.append((x, y, z))
        action_log.append('seed')
        arr = state['arr']
        intensities = [arr[s] for s in seeds]
        print(f'Seed ({x},{y},{z}) intensity={arr[x,y,z]:.1f} — '
              f'range [{min(intensities):.1f}, {max(intensities):.1f}] ({len(seeds)} seeds)')
        _resegment()

    def on_cut(x, y, z):
        r = viewer.cut_radius
        cuts.append((x, y, z, r, viewer.slice_axis))
        action_log.append('cut')
        print(f'Cut ({x},{y},{z}) radius={r} slice_axis={viewer.slice_axis} — {len(cuts)} cuts')
        _resegment()

    def on_bridge(start, end):
        r = viewer.bridge_radius
        print(f'Bridging {start} → {end} radius={r}...')
        b = bridge_path(state['arr'], start, end, radius=r)
        n_total = int(b.sum())
        slices_hit = np.where(b.any(axis=tuple(viewer.display_axes)))[0]
        print(f'Bridge: {n_total} voxels across slices {slices_hit[[0, -1]].tolist()} '
              f'(currently viewing slice {viewer.current_slice})')
        bridges.append(b)
        action_log.append('bridge')
        _resegment()

    def on_undo():
        if not action_log:
            return
        action = action_log.pop()
        if action == 'seed' and seeds:
            seeds.pop()
            print(f'Removed last seed, {len(seeds)} remaining')
        elif action == 'cut' and cuts:
            cuts.pop()
            print(f'Removed last cut, {len(cuts)} remaining')
        elif action == 'bridge' and bridges:
            bridges.pop()
            print(f'Removed last bridge, {len(bridges)} remaining')
        _resegment()

    def on_save():
        img = state['nib_img']
        default = os.path.join(state['dir'], 'segmentation.nii.gz')
        path, _ = QFileDialog.getSaveFileName(
            viewer, 'Save Segmentation', default, 'NIfTI (*.nii.gz *.nii)')
        if path:
            nib.save(make_seg_nifti(state['mask'], img.affine, img.header), path)
            print(f'Saved: {path}')

    def on_load():
        path, _ = QFileDialog.getOpenFileName(
            viewer, 'Open NIfTI', '', 'NIfTI (*.nii.gz *.nii)')
        if not path:
            return
        _load_nifti(path)
        _reset_session()
        viewer.reset(norm255(state['arr']), np.zeros_like(state['arr']))
        viewer.setWindowTitle(f'Segment — {path}')

    # Initial file selection — exit cleanly if cancelled
    path, _ = QFileDialog.getOpenFileName(
        None, 'Open NIfTI', '', 'NIfTI (*.nii.gz *.nii)')
    if not path:
        sys.exit(0)

    _load_nifti(path)

    viewer = VolumeViewer(
        norm255(state['arr']), np.zeros_like(state['arr']),
        title=f'Segment — {path}',
        seed_callback=on_click,
        undo_callback=on_undo,
        cut_callback=on_cut,
        bridge_callback=on_bridge,
        expand_to_max_callback=_resegment,
        save_callback=on_save,
        load_callback=on_load,
    )
    viewer.show()
    sys.exit(app.exec_())
