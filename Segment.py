import os
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
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


def _erase_cube(mask, centre, radius):
    """Zero out a cubic region of voxels in mask around centre (in-place)."""
    cx, cy, cz = (int(round(c)) for c in centre)
    r = radius
    x0, x1 = max(0, cx - r), min(mask.shape[0], cx + r + 1)
    y0, y1 = max(0, cy - r), min(mask.shape[1], cy + r + 1)
    z0, z1 = max(0, cz - r), min(mask.shape[2], cz + r + 1)
    mask[x0:x1, y0:y1, z0:z1] = False


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
        for cx, cy, cz, r in cuts:
            _erase_cube(threshold_mask, (cx, cy, cz), r)
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
        """Clear all seeds, cuts, and mask for a fresh segmentation."""
        seeds.clear()
        cuts.clear()
        action_log.clear()
        state['mask'][:] = False

    def _resegment():
        hi = float(np.max(state['arr'])) if viewer.expand_to_max else None
        state['mask'][:] = segment_from_seeds(
            state['arr'], seeds, cuts=cuts, hi_bound=hi)
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
        cuts.append((x, y, z, r))
        action_log.append('cut')
        print(f'Cut ({x},{y},{z}) radius={r} — {len(cuts)} cuts')
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
        expand_to_max_callback=_resegment,
        save_callback=on_save,
        load_callback=on_load,
    )
    viewer.show()
    sys.exit(app.exec_())
