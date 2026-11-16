import sys
import os
import re
import time
from collections import Counter
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from scipy.ndimage import label as ndi_label, find_objects
import PyQt5.sip
sys.modules.setdefault('sip', PyQt5.sip)  # landmark .npy files pickle QPointF objects, whose
                                            # unpickler does a bare `import sip`, not `PyQt5.sip`
from PIL import Image
import skimage as ski
from TileUtils import make_tissue_mask, load_landmarks, get_bf_slice_index, CSV_CZI_lookup
from RabbitPathFinder import find_all_the_paths
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize

# Rescoped from TilestoVoxels.py to handle ONLY voxel-boundary tracing (per-ID
# pixel-grid boundary tracing in label space -> spline-project into H&E space) and
# save the resulting polygon geometry. All feature extraction (MR
# intensity sampling, H&E texture features, full-res CZI/TIFF reading) has moved
# out to other scripts, which is why this no longer touches pyvips/CziFile/pandas/
# HnEFeatureExtraction — those were only needed for feature extraction, not tracing.
# TilestoVoxels.py (the original) still has the full pipeline if needed.


def _group_polys_by_id(poly_ids, bf_polys):
    id_to_polys = {}
    for pid, poly in zip(poly_ids, bf_polys):
        id_to_polys.setdefault(pid, []).append(poly)
    return id_to_polys


def plot_duplicate_overview(label_slice_us, tile_mask, dupes, already_split_ids, single_component_ids,
                             slice_label=''):
    """Grayscale voxel-label background, H&E tissue-mask boundary as a white
    reference line, and each duplicated voxel's ground-truth pixels
    (label_slice_us == vid) highlighted — not a re-rasterization of the traced
    polygon(s): cv2.fillPoly needs integer pixel coordinates, and our polygons'
    vertices sit at exact half-integer pixel-corner coordinates, so truncating them
    would badly over-fill (confirmed by direct test). Voxel IDs already split in
    the raw label array itself (a real gap, from summarize_ground_truth_connectivity)
    get their own rainbow color — those are worth chasing. Voxel IDs that are
    single-component in the raw label array (an unavoidable corner-touch, or a
    tissue-mask cut) are shown in black — nothing to fix in the tracing itself."""
    if not dupes:
        return
    base = label_slice_us.astype(float)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    rgb = np.stack([base, base, base], axis=-1)

    for vid in single_component_ids:
        rgb[label_slice_us == vid] = [0.0, 0.0, 0.0]

    cmap = plt.cm.gist_rainbow
    for i, vid in enumerate(already_split_ids):
        color = np.array(cmap(i / max(1, len(already_split_ids) - 1))[:3])
        rgb[label_slice_us == vid] = color

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    mask_contours, _ = cv2.findContours(tile_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in mask_contours:
        pts = c[:, 0, :]
        ax.plot(pts[:, 0], pts[:, 1], '-', color='white', linewidth=1, alpha=0.7)
    ax.set_title(f'{len(dupes)} duplicated voxel IDs — rainbow={len(already_split_ids)} already split in '
                 f'raw array, black={len(single_component_ids)} corner-touch/tissue-mask cut — '
                 f'grayscale=voxel labels, white=H&E tissue mask boundary{slice_label}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_duplicate_examples(label_slice_us, tile_mask, dupes, poly_ids, bf_polys, n_examples=9, pad=15):
    """Zoomed-in crops around a handful of duplicated voxel IDs. Magenta fill is
    ground truth (label_slice_us == vid) — same cv2.fillPoly/int32-truncation
    padding issue as plot_duplicate_overview otherwise. Green outline is each
    fragment's own traced polygon geometry (accurate float coordinates), tissue-mask
    boundary overlaid, to inspect individual split cases up close."""
    if not dupes:
        return
    id_to_polys = _group_polys_by_id(poly_ids, bf_polys)
    ids_to_show = list(dupes.keys())[:n_examples]
    ncols = min(3, len(ids_to_show))
    nrows = int(np.ceil(len(ids_to_show) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, vid in zip(axes, ids_to_show):
        polys = id_to_polys[vid]
        all_x = np.concatenate([np.array(p.exterior.coords)[:, 0] for p in polys])
        all_y = np.concatenate([np.array(p.exterior.coords)[:, 1] for p in polys])
        y0, y1 = max(0, int(all_y.min()) - pad), min(label_slice_us.shape[0], int(all_y.max()) + pad)
        x0, x1 = max(0, int(all_x.min()) - pad), min(label_slice_us.shape[1], int(all_x.max()) + pad)

        crop_label = label_slice_us[y0:y1, x0:x1]
        crop_mask = tile_mask[y0:y1, x0:x1]
        base = crop_label.astype(float)
        base = (base - base.min()) / (base.max() - base.min() + 1e-9)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[crop_label == vid] = [1.0, 0.0, 1.0]
        ax.imshow(rgb)

        # Exact per-fragment boundary (traced geometry, accurate float coordinates),
        # drawn as a line so touching/overlapping/gapped edges are visible precisely,
        # not just inferred from the ground-truth fill.
        for poly in polys:
            verts = np.array(poly.exterior.coords) - [x0, y0]
            ax.plot(verts[:, 0], verts[:, 1], '-', color='lime', linewidth=1.2)

        mask_contours, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in mask_contours:
            pts = c[:, 0, :]
            ax.plot(pts[:, 0], pts[:, 1], '-', color='cyan', linewidth=1)
        ax.set_title(f'voxel_id={vid} ({dupes[vid]} fragments, traced geometry)', fontsize=9)
        ax.axis('off')
    for ax in axes[len(ids_to_show):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def summarize_ground_truth_connectivity(dupes, label_slice_us, id_label='duplicated'):
    """For each given voxel ID, checks whether it's already disconnected in the raw
    label array itself (a real gap the registration/label volume put there) vs.
    single-component (8-connected) in the raw label array despite still coming out
    as 2+ separate traced polygons. The latter isn't a tracing bug: it's either a
    single-pixel corner touch (no simple polygon can represent that as one ring —
    see polygon_from_mask) or the tissue mask cutting through an otherwise-connected
    blob, not something to "fix" in the tracing step itself.

    Returns (already_split_ids, single_component_ids) so callers (e.g.
    plot_duplicate_overview) can distinguish the two without recomputing the same
    connectivity check."""
    already_split_ids, single_component_ids = [], []
    for vid in dupes:
        id_mask = (label_slice_us == vid)
        _, n_components = ndi_label(id_mask, structure=np.ones((3, 3)))
        if n_components > 1:
            already_split_ids.append(vid)
        else:
            single_component_ids.append(vid)
    print(f"  Of {len(dupes)} {id_label} voxel ID(s): {len(already_split_ids)} already split in the raw "
          f"label array itself, {len(single_component_ids)} single-component in the raw label "
          f"array (corner-touch or tissue-mask cut) but still traced as separate polygons")
    return already_split_ids, single_component_ids


def downsample_mask_or(mask, target_shape):
    """OR-pools a boolean mask down from H&E's 4x-upsampled resolution to label_arr's
    native per-slice resolution -- the exact inverse of the 4x np.repeat + crop that
    built label_slice_us from label_slice. Pads up to a clean 4x(target_shape) first
    (label_slice_us can be smaller than the full 4x upsample because of that trailing
    crop) with False, so padding never manufactures extra tissue. OR (not majority/AND)
    stays permissive -- a block counts as tissue if ANY of its 16 source pixels did --
    to avoid eroding thin real bridges any more than necessary."""
    th, tw = target_shape
    h, w = mask.shape
    full_h, full_w = th * 4, tw * 4
    padded = np.zeros((full_h, full_w), dtype=bool)
    padded[:min(h, full_h), :min(w, full_w)] = mask[:min(h, full_h), :min(w, full_w)]
    return padded.reshape(th, 4, tw, 4).any(axis=(1, 3))


def summarize_3d_connectivity(dupe_ids, label_arr, id_label='duplicated'):
    """Same idea as summarize_ground_truth_connectivity, but checks true 3D
    connectivity (26-connected, structure=np.ones((3,3,3))) in the full registered
    BlockFace-space label volume, instead of connectivity within a single 2D slice.
    A voxel that looks split into 2+ pieces in one cross-section can still be one
    smooth, fully-connected 3D blob — the same way a diagonal slice near the edge
    of a sphere can look like two separate blobs in 2D even though the sphere
    itself is obviously one piece. Returns (connected_in_3d_ids, disconnected_in_3d_ids).

    Uses find_objects to get every ID's 3D bounding box in a single pass over the
    whole volume, then runs the connectivity check on just that local crop — doing
    `label_arr == vid` + a full-volume ndi_label per ID (one full pass over a
    multi-million-voxel array per flagged ID) is what made this prohibitively slow.
    Unlike trace_by_id's per-slice version of this same trap, no compact-ID remap
    is needed here since find_objects only runs once for the whole check, not once
    per slide.
    """
    label_int_3d = label_arr.astype(np.int64)
    objs = find_objects(label_int_3d)

    connected_in_3d_ids, disconnected_in_3d_ids = [], []
    for vid in dupe_ids:
        sl = objs[vid - 1] if vid - 1 < len(objs) else None
        if sl is None:
            continue  # ID not present in this (registered/warped) volume at all
        local_mask = label_int_3d[sl] == vid
        _, n_components = ndi_label(local_mask, structure=np.ones((3, 3, 3)))
        if n_components == 1:
            connected_in_3d_ids.append(vid)
        else:
            disconnected_in_3d_ids.append(vid)
    print(f"  Of {len(dupe_ids)} {id_label} voxel ID(s) (flagged from 2D slices): "
          f"{len(connected_in_3d_ids)} are actually one connected piece in 3D (a slicing-plane "
          f"artifact, not a real split), {len(disconnected_in_3d_ids)} are genuinely disconnected in 3D too")
    return connected_in_3d_ids, disconnected_in_3d_ids


def polygon_from_mask(mask):
    """Traces a binary mask's exact pixel-grid boundary. Each foreground pixel is
    treated as its own unit square spanning index ± 0.5 (matching how imshow
    actually renders a pixel cell) — for every foreground pixel, keep whichever of
    its 4 unit-square edges border a background pixel or the mask's edge, then
    stitch that whole set of edge segments into closed ring(s) via
    shapely.ops.polygonize.

    This replaces feeding cv2.findContours' pixel-index output straight into
    Polygon() as if the indices were vertex coordinates: that treats a pixel index
    as a point instead of the center of a unit square, so a W x H block's raw
    contour polygon comes out (W-1) x (H-1) — short by exactly one pixel of width
    and height — and anything only 1px wide in some direction collapses to a
    degenerate zero-area contour (confirmed: a single isolated pixel returns just
    one contour point, not even a valid polygon). Building from unit-square edges
    instead reproduces the mask's true rasterized footprint exactly, so no buffer
    or rectilinearize is needed to compensate.

    Doesn't special-case a mask with a genuine hole in it (background fully
    enclosed by foreground) — rare enough for a single voxel's footprint that it
    isn't handled specially yet; polygonize would return the hole as its own small
    polygon rather than subtracting it from the outer ring.
    """
    H, W = mask.shape
    segments = []
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        if r == 0 or not mask[r - 1, c]:
            segments.append(((c - 0.5, r - 0.5), (c + 0.5, r - 0.5)))  # top
        if r == H - 1 or not mask[r + 1, c]:
            segments.append(((c - 0.5, r + 0.5), (c + 0.5, r + 0.5)))  # bottom
        if c == 0 or not mask[r, c - 1]:
            segments.append(((c - 0.5, r - 0.5), (c - 0.5, r + 0.5)))  # left
        if c == W - 1 or not mask[r, c + 1]:
            segments.append(((c + 0.5, r - 0.5), (c + 0.5, r + 0.5)))  # right
    if not segments:
        return []
    lines = unary_union([LineString(seg) for seg in segments])
    return list(polygonize(lines))


def trace_by_id(label_slice_us, tile_mask, splines, pad=3):
    """Traces each voxel ID's own binary mask directly (via polygon_from_mask),
    instead of inferring boundaries from a gradient-sign edge_map (the old
    approach, still in TilestoVoxels.py). The boundary between two different IDs
    is exactly the pixel interface here, so there's no central-difference erosion
    (np.gradient looks two pixels apart) that can manufacture a split in a thin
    region that isn't actually disconnected in the label array.

    IDs are globally unique across the whole rabbit, not per-slice, so running
    find_objects against the raw label values would allocate a bounding-box slot
    for every ID up to the largest one present in this slice — mostly empty, since
    only a few thousand of them actually occur here. Remapped to a locally-compact
    1..k range first (k = IDs actually present in this slice) so find_objects' cost
    tracks this slice's voxel count instead of the rabbit's global ID range.
    """
    label_int = label_slice_us.astype(np.int64)
    fg = label_int > 0
    present_ids = np.unique(label_int[fg])
    if present_ids.size == 0:
        return [], [], [], [], 0

    compact = np.zeros(label_int.shape, dtype=np.int64)
    compact[fg] = np.searchsorted(present_ids, label_int[fg]) + 1  # 1..k, order matches present_ids

    objs = find_objects(compact)
    H, W = label_slice_us.shape

    hne_polys, bf_polys, poly_ids, bf_centroids = [], [], [], []
    n_invalid = 0
    for idx, sl in enumerate(objs):
        if sl is None:
            continue
        vid = int(present_ids[idx])
        #Crop to region surrounding vox ID-
        y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)

        crop_compact = compact[y0:y1, x0:x1]
        crop_tile = tile_mask[y0:y1, x0:x1]

        # crop_compact == idx+1 already implies label>0 (compact is 0 wherever the
        # source label wasn't foreground), so no separate valid_mask check needed here.
        id_mask = (crop_compact == idx + 1) & crop_tile.astype(bool)
        if not id_mask.any():
            continue

        for poly in polygon_from_mask(id_mask):
            pts = np.array(poly.exterior.coords) + np.array([x0, y0])  # local crop -> full-slice coords
            poly = Polygon(pts)

            bf_pts = np.array(poly.exterior.coords)
            #Polygon transformed back to full res HnE space->
            hne_pts = splines(bf_pts[:, ::-1])
            hne_poly = Polygon(hne_pts)
            if not hne_poly.is_valid:
                hne_poly = hne_poly.buffer(0)
                if hne_poly.geom_type == 'MultiPolygon':
                    hne_poly = max(hne_poly.geoms, key=lambda g: g.area)
                if not hne_poly.is_valid:
                    n_invalid += 1
                    continue

            hne_polys.append(hne_poly)
            bf_polys.append(poly)
            poly_ids.append(vid)
            bf_centroids.append((int(poly.centroid.y), int(poly.centroid.x)))
    return poly_ids, bf_polys, hne_polys, bf_centroids, n_invalid


def plot_trace_construction(vid, label_slice_us, tile_mask, pad=6):
    """Zoomed view of how trace_by_id builds one voxel's polygon(s) via
    polygon_from_mask: the raw pixel mask side by side with the traced polygon(s)
    on the same pixel grid. The overlay should land exactly on the true pixel-cell
    edges (not through pixel centers, no buffer needed) — total polygon area is
    annotated against the true pixel count as a concrete check.
    Mirrors trace_by_id's per-ID logic directly (by value, not the compact-ID
    remap) since this only ever runs on one voxel at a time."""
    coords = np.argwhere(label_slice_us == vid)
    if coords.size == 0:
        print(f"voxel_id={vid} not present in this slice")
        return
    y0, y1 = max(0, coords[:, 0].min() - pad), min(label_slice_us.shape[0], coords[:, 0].max() + pad + 1)
    x0, x1 = max(0, coords[:, 1].min() - pad), min(label_slice_us.shape[1], coords[:, 1].max() + pad + 1)

    crop_label = label_slice_us[y0:y1, x0:x1]
    crop_tile = tile_mask[y0:y1, x0:x1]
    id_mask = (crop_label == vid) & crop_tile.astype(bool)

    polys = polygon_from_mask(id_mask)
    true_pixel_count = int(id_mask.sum())
    total_poly_area = sum(p.area for p in polys)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, title in zip(axes, ['Raw id_mask (pixel-exact)', 'Traced polygon(s) (polygon_from_mask)']):
        ax.imshow(id_mask, cmap='gray', interpolation='nearest')
        ax.set_xticks(np.arange(-0.5, id_mask.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, id_mask.shape[0], 1), minor=True)
        ax.grid(which='minor', color='cyan', linewidth=0.4, alpha=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=9)

    for poly in polys:
        verts = np.array(poly.exterior.coords)
        axes[1].plot(verts[:, 0], verts[:, 1], '-o', color='yellow', linewidth=1.5, markersize=4)

    plt.suptitle(f'voxel_id={vid} — {len(polys)} polygon(s), traced area={total_poly_area:.0f} '
                 f'vs. true pixel count={true_pixel_count}')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    Rabbit = 'R24-058'
    block_no = 6
    Block = f'Block{block_no:02d}'
    CEPH_BASE = '/System/Volumes/Data/ceph/hifu'
    root_dir = os.path.join(CEPH_BASE, 'users/jbonaventura/RabbitRegistrationProj/RabbitData')
    rabbase = os.path.join(root_dir, Rabbit)
    hne_base_dir = os.path.join(rabbase, 'HnE', Block)

    reg_HnE_dir = os.path.join(hne_base_dir, 'Registered')
    # Unique-ID label volume from MakeVoxelVolV2.py, already registered to BlockFace.
    # Requires MakeVoxelVolV2.py to have been run for this same Rabbit/Block first.
    reg_show = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps',
                             f'voxel_tile_vol_ComposedRegTo{Block}.nii.gz')

    bf_cropped_dir = os.path.join(rabbase, 'BlockFace_RGB', Block, 'CroppedImages')
    print(bf_cropped_dir)

    label_nib = nib.load(reg_show)
    #Array of registered voxel volume->
    label_arr = label_nib.get_fdata()
    #Correct axis orientation misalignment->
    if label_nib.affine[0, 0] > 0 and label_nib.affine[1, 1] > 0:
        label_arr = label_arr[::-1, ::-1, :]

    #Grab registered H&E files-
    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir) if f.lower().endswith('reg.png') and not f.startswith('._'))
    #Get blockface Indicies from H&E names-
    hne_bf_indices = [re.search(r'\d+', f).group()[-2:] for f in hne_filenames]
    #Stack H&E files into an array-
    hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in hne_filenames]
    reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)

    print(reg_HnE_arr.shape)
    print(label_arr.shape)
    print(hne_bf_indices[0])

    Show_Plots = False
    Show_Duplicate_Diagnostics = False
    scale_fac = 20  # scale factor between HnE raw and downsampled
    geoms_dir = os.path.join(rabbase, 'Analysis', Block)
    os.makedirs(geoms_dir, exist_ok=True)

    all_dupe_ids_seen = set()  # every voxel ID flagged as 2D-duplicated on any slide,
                                # for a single 3D connectivity pass once the loop ends

    # --- InVivo slide-map sanity check: paint the ORIGINAL InVivo MR volume by
    # which H&E slide each voxel ended up on-
    Build_InVivo_Slide_Map =False
    if Build_InVivo_Slide_Map:
        invivo_paths = find_all_the_paths(Rabbit, block_no, root_dir, 'InVivo')
        invivo_ref_nib = nib.load(str(invivo_paths['Moving_FilePath']))
        voxel_maps_dir = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps')
        lookup_path = os.path.join(voxel_maps_dir, 'voxel_tile_vol_id_lookup.npy')
        id_lookup = np.load(lookup_path)  # id_lookup[id-1] = (i,j,k) in InVivo space
        invivo_slide_map = np.zeros(invivo_ref_nib.shape[:3], dtype=np.float32)
        invivo_paint_count = np.zeros(invivo_ref_nib.shape[:3], dtype=np.int32)
        n_slide_map_conflicts = 0
        conflict_slice_distances = []  # |slice_num - previous slice_num| at each conflict

    for img in range(len(hne_filenames)):
        hne_ds_im = reg_HnE_arr[:, :, img, :]
        hne_Name = hne_bf_indices[img]

        # Finding tps transform between CZI and blockface -
        img_number = re.search(r'\d+', hne_filenames[img]).group()  # e.g. '0011'
        landmarks = load_landmarks(hne_base_dir, img_number)
        #Find CZI file-
        CZI_filepath = CSV_CZI_lookup(Rabbit, Block, img_number, ceph_base=CEPH_BASE)
        czi_stem = os.path.splitext(os.path.basename(CZI_filepath))[0]

        #Sort Landmarks and reform Splines transform->
        src = np.array([[p.y(), p.x()] for p in landmarks[0]]) * 3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]]) * 3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)

        # Use blockface filename position in CroppedImages to find the correct NIfTI slice -
        slice_num = get_bf_slice_index(bf_cropped_dir, img_number)
        print(slice_num)
        #Pull Slice from voxel vol corresponding to HnE slide->
        label_slice = label_arr[:, :, slice_num].T
        print("label_slice info", np.max(label_slice), np.min(label_slice))
        # Upsampling by 4 to pull to registered HNE resolution-
        label_slice_us = np.repeat(np.repeat(label_slice, 4, axis=0), 4, axis=1)
        label_slice_us = label_slice_us[:hne_ds_im.shape[0], :hne_ds_im.shape[1]]

        print(hne_ds_im.shape, label_slice_us.shape)

        #Mask tissue area in HnE slide->
        tile_mask = make_tissue_mask(reg_HnE_arr[:, :, img, :])

        if Show_Plots:
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
            axes[0].imshow(hne_ds_im)
            axes[0].set_title('H&E', fontsize=8)
            axes[0].axis('off')
            axes[1].imshow(hne_ds_im * tile_mask[:, :, np.newaxis])
            axes[1].set_title('Tissue mask applied', fontsize=8)
            axes[1].axis('off')
            axes[2].imshow(label_slice_us, cmap='Grays')
            axes[2].set_title('Voxel labels', fontsize=8)
            axes[2].axis('off')
            plt.suptitle(f'Slice {hne_Name}')
            plt.tight_layout(pad=1)
            plt.show()
            plt.close()

        # --- Trace voxel boundaries directly per voxel ID (see trace_by_id) instead
        # of the old global gradient/edge_map approach (still in TilestoVoxels.py if
        # ever needed again). ---
        t0 = time.perf_counter()
        poly_ids, bf_polys, hne_polys, bf_centroids, n_invalid = trace_by_id(
            label_slice_us, tile_mask, splines
        )
        elapsed = time.perf_counter() - t0
        print(f"Slice {img}: {len(hne_polys)} valid polygons, {n_invalid} dropped as invalid ({elapsed:.1f}s)")

        # --- Check for voxel IDs traced as multiple fragments.
        id_counts = Counter(poly_ids)
        dupes = {vid: n for vid, n in id_counts.items() if n > 1}
        if dupes:
            all_dupe_ids_seen.update(dupes.keys())
            preview = list(dupes.items())[:10]
            print(f"  WARNING: {len(dupes)} voxel ID(s) traced as multiple fragments: "
                  f"{preview}{' ...' if len(dupes) > 10 else ''}")
            already_split_ids, single_component_ids = summarize_ground_truth_connectivity(dupes, label_slice_us)
            if Show_Duplicate_Diagnostics:
                plot_duplicate_overview(label_slice_us, tile_mask, dupes, already_split_ids, single_component_ids,
                                         slice_label=f' (slice {hne_Name})')
                plot_duplicate_examples(label_slice_us, tile_mask, dupes, poly_ids, bf_polys)

        # --- Paint this slide's voxel IDs into the InVivo slide map-
        if Build_InVivo_Slide_Map:
            slide_value = (slice_num + 1) * 100
            for vid in set(poly_ids):
                i, j, k = id_lookup[vid - 1]
                if invivo_paint_count[i, j, k] > 0:
                    prev_slice_num = int(invivo_slide_map[i, j, k]) // 100 - 1
                    conflict_slice_distances.append(abs(slice_num - prev_slice_num))
                    n_slide_map_conflicts += 1
                invivo_slide_map[i, j, k] = slide_value
                invivo_paint_count[i, j, k] += 1

        # --- Save polygon geometry (H&E space). poly_id == the voxel's unique label,
        # decodable back to InVivo (i,j,k) via MakeVoxelVolV2.py's id_lookup.npy —
        # no separate lookup/bookkeeping needed in this script or its consumers. ---
        geojson_features = []
        for pid, poly, (centroid_row, centroid_col) in zip(poly_ids, hne_polys, bf_centroids):
            coords = np.array(poly.exterior.coords)
            coords[:, 0] = coords[:, 0] * scale_fac
            coords[:, 1] = coords[:, 1] * scale_fac
            ring = [[float(x), float(y)] for x, y in coords]
            geojson_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "object_type": "annotation",
                    "name": f"voxel_{pid}",
                    "poly_id": int(pid),
                    "centroid_row": int(centroid_row),
                    "centroid_col": int(centroid_col),
                    "slice_num": int(slice_num),
                }
            })
        geojson_path = os.path.join(geoms_dir, f'voxel_polygons_{czi_stem}.geojson')
        with open(geojson_path, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": geojson_features}, f)
        print(f"Saved {len(geojson_features)} polygons to {geojson_path}")

    if Build_InVivo_Slide_Map:
        slide_map_path = os.path.join(geoms_dir, 'invivo_slide_map.nii.gz')
        nib.save(nib.Nifti1Image(invivo_slide_map, invivo_ref_nib.affine, invivo_ref_nib.header), slide_map_path)
        n_painted = int((invivo_paint_count > 0).sum())
        print(f"Saved InVivo slide-map volume to {slide_map_path}")
        print(f"  {n_slide_map_conflicts} voxel(s) painted by more than one slide out of "
              f"{n_painted} painted voxel(s) total")

        # BlockFace's own Z spacing converts slice_num distance to physical mm, so
        # conflicts can be checked directly against the ~0.5mm (MR voxel size) /
        # 0.25mm (section spacing) overlap expected from geometry alone, rather than
        # just eyeballing an index count.
        bf_z_spacing_mm = float(label_nib.header.get_zooms()[2])
        dist_counts = Counter(conflict_slice_distances)
        print(f"  BlockFace Z spacing ~= {bf_z_spacing_mm:.3f} mm/slice_num-step")
        for d, n in sorted(dist_counts.items()):
            print(f"    slice_num distance={d} (~{d * bf_z_spacing_mm:.3f} mm): {n} conflict(s)")

    # --- 3D connectivity check: every voxel ID flagged as split within some 2D
    # slice, re-checked against the full registered 3D label volume. A split that
    # disappears in 3D was a slicing-plane artifact (the cross-section cut through
    # a thin waist of an otherwise-connected shape), not a real gap. ---
    if all_dupe_ids_seen:
        print(f"3D connectivity check across {len(all_dupe_ids_seen)} voxel ID(s) flagged as split in some slice:")
        connected_in_3d_ids, disconnected_in_3d_ids = summarize_3d_connectivity(all_dupe_ids_seen, label_arr)
