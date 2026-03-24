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
    registered_path = '/Users/jbonaventura/Downloads/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToExVivo_0304-1231.nii.gz'
    fixed_path = '/Users/jbonaventura/Downloads/RabbitData/R23-055/ExVivo_MR/ExVivo.nii'

    reg_img = nib.load(registered_path)
    reg_arr = reg_img.get_fdata()

    fixed_img = nib.load(fixed_path)
    fixed_can = nib.as_closest_canonical(fixed_img)
    fixed_arr = fixed_can.get_fdata()
    zooms = fixed_can.header.get_zooms()

    print('Applying N4 bias field correction...')
    reg_arr = flatten_field(reg_arr)
    fixed_arr = flatten_field(fixed_arr)
    print('Done.')

    reg_seeds = []
    reg_cuts = []
    reg_action_log = []   # 'seed' or 'cut' entries in placement order
    fixed_seeds = []
    fixed_cuts = []
    fixed_action_log = []
    reg_mask = np.zeros_like(reg_arr, dtype=bool)
    fixed_mask = np.zeros_like(fixed_arr, dtype=bool)

    def _resegment_reg():
        hi = float(np.max(reg_arr)) if viewer_reg.expand_to_max else None
        reg_mask[:] = segment_from_seeds(reg_arr, reg_seeds, cuts=reg_cuts, hi_bound=hi)
        viewer_reg.data2 = reg_mask.astype(float) * 255
        viewer_reg.update_plot()

    def _resegment_fixed():
        hi = float(np.max(fixed_arr)) if viewer_fixed.expand_to_max else None
        fixed_mask[:] = segment_from_seeds(fixed_arr, fixed_seeds, cuts=fixed_cuts, hi_bound=hi)
        viewer_fixed.data2 = fixed_mask.astype(float) * 255
        viewer_fixed.update_plot()

    def on_reg_click(x, y, z):
        reg_seeds.append((x, y, z))
        reg_action_log.append('seed')
        intensities = [reg_arr[s] for s in reg_seeds]
        print(f'Registered seed ({x},{y},{z}) intensity={reg_arr[x,y,z]:.1f} — range [{min(intensities):.1f}, {max(intensities):.1f}] ({len(reg_seeds)} seeds)')
        _resegment_reg()

    def on_reg_cut(x, y, z):
        r = viewer_reg.cut_radius
        reg_cuts.append((x, y, z, r))
        reg_action_log.append('cut')
        print(f'Registered cut ({x},{y},{z}) radius={r} — {len(reg_cuts)} cuts')
        _resegment_reg()

    def on_reg_undo():
        if not reg_action_log:
            return
        action = reg_action_log.pop()
        if action == 'seed' and reg_seeds:
            reg_seeds.pop()
            print(f'Registered: removed last seed, {len(reg_seeds)} remaining')
        elif action == 'cut' and reg_cuts:
            reg_cuts.pop()
            print(f'Registered: removed last cut, {len(reg_cuts)} remaining')
        _resegment_reg()

    def on_fixed_click(x, y, z):
        fixed_seeds.append((x, y, z))
        fixed_action_log.append('seed')
        intensities = [fixed_arr[s] for s in fixed_seeds]
        print(f'Fixed seed ({x},{y},{z}) intensity={fixed_arr[x,y,z]:.1f} — range [{min(intensities):.1f}, {max(intensities):.1f}] ({len(fixed_seeds)} seeds)')
        _resegment_fixed()

    def on_fixed_cut(x, y, z):
        r = viewer_fixed.cut_radius
        fixed_cuts.append((x, y, z, r))
        fixed_action_log.append('cut')
        print(f'Fixed cut ({x},{y},{z}) radius={r} — {len(fixed_cuts)} cuts')
        _resegment_fixed()

    def on_fixed_undo():
        if not fixed_action_log:
            return
        action = fixed_action_log.pop()
        if action == 'seed' and fixed_seeds:
            fixed_seeds.pop()
            print(f'Fixed: removed last seed, {len(fixed_seeds)} remaining')
        elif action == 'cut' and fixed_cuts:
            fixed_cuts.pop()
            print(f'Fixed: removed last cut, {len(fixed_cuts)} remaining')
        _resegment_fixed()

    def on_reg_save():
        path, _ = QFileDialog.getSaveFileName(
            viewer_reg, 'Save Registered Segmentation', 'reg_segmentation.nii.gz',
            'NIfTI (*.nii.gz *.nii)')
        if path:
            nib.save(make_seg_nifti(reg_mask, reg_img.affine, reg_img.header), path)
            print(f'Saved: {path}')

    def on_fixed_save():
        path, _ = QFileDialog.getSaveFileName(
            viewer_fixed, 'Save Fixed Segmentation', 'fixed_segmentation.nii.gz',
            'NIfTI (*.nii.gz *.nii)')
        if path:
            nib.save(make_seg_nifti(fixed_mask, fixed_can.affine, fixed_can.header), path)
            print(f'Saved: {path}')

    app = QApplication(sys.argv)
    viewer_reg = VolumeViewer(norm255(reg_arr), np.zeros_like(reg_arr),
                              'Registered InVivo — click inside ablation to seed',
                              seed_callback=on_reg_click, undo_callback=on_reg_undo,
                              cut_callback=on_reg_cut,
                              expand_to_max_callback=_resegment_reg,
                              save_callback=on_reg_save)
    viewer_reg.show()
    viewer_fixed = VolumeViewer(norm255(fixed_arr), np.zeros_like(fixed_arr),
                                'Fixed ExVivo — click inside ablation to seed',
                                seed_callback=on_fixed_click, undo_callback=on_fixed_undo,
                                cut_callback=on_fixed_cut,
                                expand_to_max_callback=_resegment_fixed,
                                save_callback=on_fixed_save)
    viewer_fixed.move(820, 100)
    viewer_fixed.show()
    sys.exit(app.exec_())
