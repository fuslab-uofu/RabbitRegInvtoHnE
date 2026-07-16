#Import Desired Libraries
import os
import re
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")  # Homebrew's arm64 lib dir isn't on
                                                                    # macOS's default dlopen search path;
                                                                    # pyvips needs it to find libvips.42.dylib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pyvips
import torchstain
from aicspylibczi import CziFile
from pylibCZIrw import czi as cziwriter

from TileUtils import make_tissue_mask, swap_channel_order, CSV_CZI_lookup

CEPH_BASE = "/System/Volumes/Data/ceph/hifu"

Ref_path = os.path.join(CEPH_BASE, "users/jbonaventura/RabbitRegistrationProj/RefMaterials/HnE_R23-055_H7_7a_StainNormRef.tif")

# Add (rabbit_id, block_no) pairs here to include them in the batch run - only slides
# that made it into that block's Registered/ folder get processed (matched by pulling
# each registered image's slice number and looking up its raw CZI via CSV_CZI_lookup,
# same convention TilesToVoxels.py already uses), so un-registered edge slices are
# skipped automatically without needing to list them separately.
RABBIT_BLOCKS = [
    # ("R24-101", 11),
    # ("R24-058", 6),
    # ("R24-240", 3),
]

def registered_czi_paths(rabbit_id, block_no, ceph_base=CEPH_BASE):
    block = f"Block{block_no:02d}"
    reg_dir = os.path.join(ceph_base, "users/jbonaventura/RabbitRegistrationProj/RabbitData",
                            rabbit_id, "HnE", block, "Registered")
    # Case-insensitive on "reg.png" (vs "Reg.png" etc.) since that convention isn't
    # strictly enforced elsewhere in the pipeline.
    reg_filenames = sorted(f for f in os.listdir(reg_dir)
                           if f.lower().endswith("reg.png") and not f.startswith("._"))
    paths = []
    for fname in reg_filenames:
        img_number = re.search(r"\d+", fname).group()
        paths.append(CSV_CZI_lookup(rabbit_id, block, img_number, ceph_base=ceph_base))
    return paths


def rabbit_block_output_dir(rabbit_id, block_no, ceph_base=CEPH_BASE):
    # Output tiffs live alongside that rabbit/block's own data (RabbitData/{rabbit}/
    # HnE/Block{NN}/NormalizedFull) rather than one shared scratch folder, so they're
    # easy to find per-slide and don't get mixed across rabbits/blocks.
    block = f"Block{block_no:02d}"
    out_dir = os.path.join(ceph_base, "users/jbonaventura/RabbitRegistrationProj/RabbitData",
                            rabbit_id, "HnE", block, "NormalizedFull")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# (rabbit_id, block_no, czi_path) - keeps the rabbit/block association per file so the
# write step below can save each output next to that slide's own data.
test_slides = [(rabbit_id, block_no, p) for rabbit_id, block_no in RABBIT_BLOCKS
               for p in registered_czi_paths(rabbit_id, block_no)]
scale_factor = 1 / 20
SHOW_PLOTS = False  # master switch for every plt.show() in the batch run below - flip
                     # to True to bring back the background-tile overlay, reconstructed-
                     # slide, roundtrip-verification, and (if also True) deep diagnostic
                     # fit/tile inspection plots for troubleshooting
BACKGROUND_BRIGHTNESS_THRESHOLD = 200  # mean RGB a "0% tissue" tile crop must clear to
                                        # count as clean white background rather than a
                                        # dark anomaly (out-of-focus corner, shadow, etc.)
NORMALIZATION_PREVIEW_SCALE = 1 / 16   # whole-slide preview used for Macenko stain-vector
                                        # estimation - matches the corrected TIFF's own
                                        # embedded 1/16 pyramid page (page=4)

# Reference slide's stain vectors, fit once at import time from its own whole-slide
# preview (not per-tile - see stain_vectors_for_slide's docstring for why).
stain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
stain_normalizer.fit(pyvips.Image.tiffload(Ref_path, page=4).numpy())


def stain_vectors_for_slide(preview_rgb):
    # Estimate this slide's own (HE, maxC) once from a whole-slide preview, reusing
    # torchstain's actual PCA/percentile implementation (name-mangled but not
    # actually private - Python doesn't enforce it) rather than reimplementing it.
    # This is the only place PCA (the numerically fragile step) ever runs - never
    # per-tile, since a background-heavy 256x256 tile has too little tissue OD
    # variance for np.linalg.eigh to converge (confirmed empirically: naive
    # torchstain normalize() crashes with "Eigenvalues did not converge" on
    # background tiles). A whole-slide preview has enough real tissue that this
    # has been reliable in testing, but isn't mathematically guaranteed for an
    # arbitrarily sparse slide - callers should handle failure, not assume success.
    HE, _, maxC = stain_normalizer._NumpyMacenkoNormalizer__compute_matrices(
        preview_rgb, Io=240, alpha=1, beta=0.15)
    return HE, maxC


def normalize_tile_fixed_source(tile_rgb, source_HE, source_maxC, Io=240, beta=0.15):
    # Same recolor math as NumpyMacenkoNormalizer.normalize()'s tail, but the
    # source HE/maxC are supplied (from stain_vectors_for_slide, once per slide)
    # rather than re-derived via PCA on this tile - see stain_vectors_for_slide.
    h, w, c = tile_rgb.shape
    flat = tile_rgb.reshape((-1, 3))
    OD, _ = stain_normalizer._NumpyMacenkoNormalizer__convert_rgb2od(flat, Io=Io, beta=beta)
    C = stain_normalizer._NumpyMacenkoNormalizer__find_concentration(OD, source_HE)

    maxC_ratio = np.divide(source_maxC, stain_normalizer.maxCRef)
    C2 = np.divide(C, maxC_ratio[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-stain_normalizer.HERef.dot(C2)))
    Inorm[Inorm > 255] = 255
    return np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)


def read_downsampled_mosaic(czifile, scale_factor):
    bbox = czifile.get_mosaic_bounding_box()
    img = czifile.read_mosaic(
        C=0, scale_factor=scale_factor,
        region=(bbox.x, bbox.y, bbox.w, bbox.h),
        background_color=(1, 1, 1),
    )[0, :, :, :]
    return swap_channel_order(img), bbox


def tile_rect_in_downsampled(tile_bbox, mosaic_bbox, scale_factor, img_shape):
    # Tile bboxes are given in full-res absolute coordinates, and the mosaic's own
    # origin is not (0,0) (can be negative) - so make the tile rect relative to the
    # mosaic bbox origin before scaling, to land in the downsampled array's pixel grid.
    x0 = int(round((tile_bbox.x - mosaic_bbox.x) * scale_factor))
    y0 = int(round((tile_bbox.y - mosaic_bbox.y) * scale_factor))
    x1 = int(round((tile_bbox.x + tile_bbox.w - mosaic_bbox.x) * scale_factor))
    y1 = int(round((tile_bbox.y + tile_bbox.h - mosaic_bbox.y) * scale_factor))
    # clip to image bounds - rounding can push edge tiles slightly outside
    x0, x1 = max(x0, 0), min(x1, img_shape[1])
    y0, y1 = max(y0, 0), min(y1, img_shape[0])
    return x0, y0, x1, y1


#Gather a few background tiles
def find_background_tiles(path, scale_factor=scale_factor):
    czifile = CziFile(path)
    img, mosaic_bbox = read_downsampled_mosaic(czifile, scale_factor)
    mask = make_tissue_mask(img)

    tile_bboxes = czifile.get_all_mosaic_tile_bounding_boxes(C=0)
    background_tiles = []
    tissue_tiles = []
    dark_anomaly_tiles = []
    for tile_info, tile_bbox in tile_bboxes.items():
        rect = tile_rect_in_downsampled(tile_bbox, mosaic_bbox, scale_factor, img.shape)
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue  # tile fell entirely outside the mosaic bbox after rounding
        tissue_fraction = mask[y0:y1, x0:x1].mean()
        mean_brightness = img[y0:y1, x0:x1].mean()
        record = {"m_index": tile_info.m_index, "rect": rect,
                  "tissue_fraction": tissue_fraction, "mean_brightness": mean_brightness}
        if tissue_fraction > 0:
            tissue_tiles.append(record)
        elif mean_brightness < BACKGROUND_BRIGHTNESS_THRESHOLD:
            # 0% tissue by the saturation-based mask, but too dark to be clean white
            # background - e.g. an out-of-focus corner or shadow, not a debris speck
            dark_anomaly_tiles.append(record)
        else:
            background_tiles.append(record)
    return img, mask, background_tiles, tissue_tiles, dark_anomaly_tiles


def show_tile(img, rect, m_index):
    x0, y0, x1, y1 = rect
    plt.figure()
    plt.imshow(img[y0:y1, x0:x1])
    plt.title(f"Tile M={m_index}")
    plt.show()


def plot_background_tile_overlay(img, background_tiles, highlight_tiles=None, title=None):
    highlight_ids = {id(t) for t in highlight_tiles} if highlight_tiles else set()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for tile in background_tiles:
        x0, y0, x1, y1 = tile["rect"]
        highlighted = id(tile) in highlight_ids
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=2 if highlighted else 1,
                                  edgecolor="yellow" if highlighted else "red",
                                  facecolor="none")
        ax.add_patch(rect)
        if highlighted:
            ax.text(x0, y0 - 2, str(tile["m_index"]), color="yellow", fontsize=8)
    ax.set_title(title)
    plt.show()


def find_boundary_background_tiles(mask, background_tiles, img_shape, margin=None):
    # Flags background tiles that have tissue within roughly one tile-width of their
    # edge, by checking the mask over each tile's rect expanded by a margin - lets us
    # inspect specifically the tiles bordering tissue, where downsampling could be
    # blurring a sharp edge, or the coarse mask could be missing thin/faint tissue.
    boundary_tiles = []
    for tile in background_tiles:
        x0, y0, x1, y1 = tile["rect"]
        tile_margin = margin if margin is not None else max(x1 - x0, y1 - y0)
        mx0 = max(x0 - tile_margin, 0)
        my0 = max(y0 - tile_margin, 0)
        mx1 = min(x1 + tile_margin, img_shape[1])
        my1 = min(y1 + tile_margin, img_shape[0])
        if mask[my0:my1, mx0:mx1].any():
            boundary_tiles.append(tile)
    return boundary_tiles


def select_sample_tiles(tiles, n=5):
    # evenly spaced picks across the list rather than the first n, so the sample
    # isn't clustered wherever the dict iteration order happened to start
    step = max(1, len(tiles) // n)
    return tiles[::step][:n]


def plot_tile_grid(img, tiles, title=None):
    cols = len(tiles)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes = np.atleast_1d(axes)
    for ax, tile in zip(axes, tiles):
        x0, y0, x1, y1 = tile["rect"]
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(f"M={tile['m_index']}")
        ax.axis("off")
    fig.suptitle(title)
    plt.show()


def load_raw_tile(czifile, m_index):
    # read_image has no scale_factor - always native/full resolution (unlike read_mosaic),
    # and reads straight from that tile's subblock, with no stitching/blending from neighbors
    tile_img, _ = czifile.read_image(M=m_index, C=0)
    return swap_channel_order(np.squeeze(tile_img))


def plot_raw_tile_grid(path, tiles, title=None):
    czifile = CziFile(path)
    cols = len(tiles)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes = np.atleast_1d(axes)
    for ax, tile in zip(axes, tiles):
        tile_img = load_raw_tile(czifile, tile["m_index"])
        ax.imshow(tile_img)
        ax.set_title(f"M={tile['m_index']}")
        ax.axis("off")
    fig.suptitle(title)
    plt.show()


def build_background_stamp(path, background_tiles):
    # Stays uint8 rather than casting to float up front - a few hundred raw tiles
    # stacked is already a meaningful chunk of memory, no need to quadruple it.
    czifile = CziFile(path)
    stack = np.stack([load_raw_tile(czifile, tile["m_index"]) for tile in background_tiles], axis=0)
    median_map = np.median(stack, axis=0)
    std_map = np.std(stack, axis=0)
    return median_map, std_map


def plot_tiles_vs_median(path, tiles, median_map, title=None):
    czifile = CziFile(path)
    cols = len(tiles)
    fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 9))
    row_labels = ["raw tile", "median map", "diff (+128 offset)"]
    for col, tile in enumerate(tiles):
        tile_img = load_raw_tile(czifile, tile["m_index"]).astype(np.int16)
        diff_display = np.clip(tile_img - median_map.astype(np.int16) + 128, 0, 255).astype(np.uint8)

        axes[0, col].imshow(tile_img.astype(np.uint8))
        axes[0, col].set_title(f"M={tile['m_index']}")
        axes[1, col].imshow(median_map.astype(np.uint8))
        axes[2, col].imshow(diff_display)

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9)

    fig.suptitle(title)
    plt.show()


def build_polynomial_design(ny, nx):
    # x, y normalized to -1..1 over the given extent purely so x^4 doesn't blow up
    # the design matrix's condition number - doesn't change what's being fit/evaluated.
    # Separable quartic per channel: Z = a1*x + a2*x^2 + a3*x^3 + a4*x^4
    #                                    + b1*y + b2*y^2 + b3*y^3 + b4*y^4 + c
    x_norm = np.linspace(-1, 1, nx)
    y_norm = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x_norm, y_norm)
    return np.stack([
        X.ravel(), X.ravel() ** 2, X.ravel() ** 3, X.ravel() ** 4,
        Y.ravel(), Y.ravel() ** 2, Y.ravel() ** 3, Y.ravel() ** 4,
        np.ones(X.size),
    ], axis=1)


def fit_flatfield_surface(median_map):
    ny, nx, n_channels = median_map.shape
    design = build_polynomial_design(ny, nx)

    fitted_surface = np.zeros_like(median_map, dtype=np.float64)
    coeffs_per_channel = []
    for c in range(n_channels):
        z = median_map[:, :, c].astype(np.float64).ravel()
        coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
        fitted_surface[:, :, c] = (design @ coeffs).reshape(ny, nx)
        coeffs_per_channel.append(coeffs)

    return fitted_surface, coeffs_per_channel


def evaluate_flatfield_surface(coeffs_per_channel, ny, nx):
    # Re-plugs the already-fit coefficients into a design matrix at a different
    # (typically downsampled) resolution - same fitted surface, just re-evaluated
    # at new grid coordinates, not a new fit.
    design = build_polynomial_design(ny, nx)
    surface = np.zeros((ny, nx, len(coeffs_per_channel)), dtype=np.float64)
    for c, coeffs in enumerate(coeffs_per_channel):
        surface[:, :, c] = (design @ coeffs).reshape(ny, nx)
    return surface


def plot_cross_sections(median_map, fitted_surface, axis="x", n_slices=3, title=None):
    # axis="x": horizontal cross-sections (intensity vs. x) at a few fixed rows (y)
    # axis="y": vertical cross-sections (intensity vs. y) at a few fixed columns (x)
    ny, nx, _ = median_map.shape
    channel_colors = ["red", "green", "blue"]
    length = ny if axis == "x" else nx
    positions = np.linspace(0, length - 1, n_slices + 2)[1:-1].astype(int)

    fig, axes = plt.subplots(1, len(positions), figsize=(5 * len(positions), 4))
    axes = np.atleast_1d(axes)
    for ax, pos in zip(axes, positions):
        for c in range(3):
            if axis == "x":
                median_line, fit_line = median_map[pos, :, c], fitted_surface[pos, :, c]
            else:
                median_line, fit_line = median_map[:, pos, c], fitted_surface[:, pos, c]
            ax.plot(median_line, color=channel_colors[c], alpha=0.5,
                    label=f"{channel_colors[c]} median" if pos == positions[0] else None)
            ax.plot(fit_line, color=channel_colors[c], linestyle="--",
                    label=f"{channel_colors[c]} fit" if pos == positions[0] else None)
        ax.set_title(f"{'row y=' if axis == 'x' else 'col x='}{pos}")
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    plt.show()


def apply_flatfield_correction(tile_img, fitted_surface, target=255):
    corrected = tile_img.astype(np.float64) / fitted_surface * target
    return np.clip(corrected, 0, 255).astype(np.uint8)


def plot_correction_comparison(path, tiles, fitted_surface, title=None):
    czifile = CziFile(path)
    cols = len(tiles)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    row_labels = ["raw tile", "corrected"]
    for col, tile in enumerate(tiles):
        raw_tile = load_raw_tile(czifile, tile["m_index"])
        corrected_tile = apply_flatfield_correction(raw_tile, fitted_surface)

        axes[0, col].imshow(raw_tile)
        axes[0, col].set_title(f"M={tile['m_index']}\nraw std={raw_tile.std():.1f}")
        axes[1, col].imshow(corrected_tile)
        axes[1, col].set_title(f"corrected std={corrected_tile.std():.1f}")

        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9)

    fig.suptitle(title)
    plt.show()


def downsample_tile(tile_img, scale_factor):
    # INTER_AREA is the standard choice for shrinking - area-averages rather than
    # naively sampling, avoiding aliasing when reducing by a large factor like 20x
    new_w = max(1, int(round(tile_img.shape[1] * scale_factor)))
    new_h = max(1, int(round(tile_img.shape[0] * scale_factor)))
    return cv2.resize(tile_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def reconstruct_corrected_slide(path, coeffs_per_channel, scale_factor=scale_factor, target=255):
    # Downsamples + corrects every tile individually, then pastes each into an output
    # canvas at its own scaled bbox position - later tiles simply overwrite earlier
    # ones in overlap regions, no blending, matching how read_mosaic itself appears
    # to place tiles (per visual inspection - no blending visible at the seams).
    czifile = CziFile(path)
    mosaic_bbox = czifile.get_mosaic_bounding_box()
    canvas_w = int(round(mosaic_bbox.w * scale_factor))
    canvas_h = int(round(mosaic_bbox.h * scale_factor))
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    correction_surface = None  # built once we know the downsampled tile size, reused after
    for tile_info, tile_bbox in czifile.get_all_mosaic_tile_bounding_boxes(C=0).items():
        raw_tile = load_raw_tile(czifile, tile_info.m_index)
        ds_tile = downsample_tile(raw_tile, scale_factor)
        if correction_surface is None or correction_surface.shape[:2] != ds_tile.shape[:2]:
            correction_surface = evaluate_flatfield_surface(coeffs_per_channel, ds_tile.shape[0], ds_tile.shape[1])
        corrected_tile = apply_flatfield_correction(ds_tile, correction_surface, target=target)

        x0 = int(round((tile_bbox.x - mosaic_bbox.x) * scale_factor))
        y0 = int(round((tile_bbox.y - mosaic_bbox.y) * scale_factor))
        x1, y1 = x0 + corrected_tile.shape[1], y0 + corrected_tile.shape[0]
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, canvas_w), min(y1, canvas_h)
        if cx1 <= cx0 or cy1 <= cy0:
            continue  # tile fell entirely outside the canvas after rounding
        canvas[cy0:cy1, cx0:cx1] = corrected_tile[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]

    return canvas


def plot_correction_math_walkthrough(median_map, fitted_surface, axis="y", position=None, target=255, title=None):
    # Walks the actual arithmetic for one 1D slice: median vs. fit, then the inverted
    # flat field (1/fit) alone, then median * inverted (should hover near 1), then
    # that result rescaled by target (should hover near target - flat, corrected).
    ny, nx, _ = median_map.shape
    channel_colors = ["red", "green", "blue"]
    if axis == "y":
        pos = position if position is not None else nx // 2
        median_slice, fit_slice = median_map[:, pos, :], fitted_surface[:, pos, :]
        slice_label = f"col x={pos}"
    else:
        pos = position if position is not None else ny // 2
        median_slice, fit_slice = median_map[pos, :, :], fitted_surface[pos, :, :]
        slice_label = f"row y={pos}"

    inverted = 1.0 / fit_slice
    product = median_slice * inverted  # = median_slice / fit_slice
    rescaled = product * target

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for c in range(3):
        axes[0].plot(median_slice[:, c], color=channel_colors[c], alpha=0.5,
                     label=f"{channel_colors[c]} median" if c == 0 else None)
        axes[0].plot(fit_slice[:, c], color=channel_colors[c], linestyle="--",
                     label=f"{channel_colors[c]} fit" if c == 0 else None)
    axes[0].set_title("median vs. fit")
    axes[0].legend(fontsize=7)

    for c in range(3):
        axes[1].plot(inverted[:, c], color=channel_colors[c])
    axes[1].set_title("inverted flat field (1/fit)")

    for c in range(3):
        axes[2].plot(product[:, c], color=channel_colors[c])
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[2].set_title("median x inverted (=median/fit)")

    for c in range(3):
        axes[3].plot(rescaled[:, c], color=channel_colors[c])
    axes[3].axhline(target, color="gray", linestyle=":", linewidth=1)
    axes[3].set_title(f"rescaled (x{target})")

    # Same y-axis on "median vs. fit" and "rescaled" so the flattening is visually
    # comparable rather than each panel auto-scaling to its own tighter range.
    shared_min = min(median_slice.min(), fit_slice.min(), rescaled.min())
    shared_max = max(median_slice.max(), fit_slice.max(), rescaled.max())
    pad = (shared_max - shared_min) * 0.05
    axes[0].set_ylim(shared_min - pad, shared_max + pad)
    axes[3].set_ylim(shared_min - pad, shared_max + pad)

    fig.suptitle((title or "") + f" - {slice_label}")
    plt.show()


def get_pixel_size_um(czifile):
    # CZI stores raw (full-res) pixel size in meters under Scaling/Items/Distance
    value = czifile.meta.find('.//Scaling/Items/Distance[@Id="X"]/Value')
    return float(value.text) * 1e6


def write_corrected_czi(path, coeffs_per_channel, output_path, scale_factor=scale_factor, target=255,
                         jpeg_quality=90):
    # Same per-tile downsample+correct as reconstruct_corrected_slide, but writes each
    # corrected tile out via pylibCZIrw at its own (scaled) bbox position instead of
    # pasting into our own canvas - real stitching then comes from read_mosaic on the
    # resulting file, same as everywhere else in this pipeline.
    # czifile = CziFile(path)
    # mosaic_bbox = czifile.get_mosaic_bounding_box()
    # out_pixel_size_um = get_pixel_size_um(czifile) / scale_factor
    #
    # correction_surface = None
    # with cziwriter.create_czi(output_path, exist_ok=True, compression_options="zstd0:ExplicitLevel=1") as writer:
    #     for tile_info, tile_bbox in czifile.get_all_mosaic_tile_bounding_boxes(C=0).items():
    #         raw_tile = load_raw_tile(czifile, tile_info.m_index)
    #         ds_tile = downsample_tile(raw_tile, scale_factor)
    #         if correction_surface is None or correction_surface.shape[:2] != ds_tile.shape[:2]:
    #             correction_surface = evaluate_flatfield_surface(coeffs_per_channel, ds_tile.shape[0], ds_tile.shape[1])
    #         corrected_tile = apply_flatfield_correction(ds_tile, correction_surface, target=target)
    #
    #         x0 = int(round((tile_bbox.x - mosaic_bbox.x) * scale_factor))
    #         y0 = int(round((tile_bbox.y - mosaic_bbox.y) * scale_factor))
    #         # pylibCZIrw always tags written output as bgr24 regardless of content, so
    #         # flip back to match the tag - everything upstream has been true RGB since
    #         # load_raw_tile/read_downsampled_mosaic corrected it on the way in.
    #         writer.write(swap_channel_order(corrected_tile), location=(x0, y0))
    #
    #     # write_metadata's docstring claims scale_x/scale_y are in um, but it actually
    #     # writes the given value straight into the XML's meters-denominated field
    #     # unconverted (verified empirically) - so convert um -> m here to compensate.
    #     writer.write_metadata(scale_x=out_pixel_size_um * 1e-6, scale_y=out_pixel_size_um * 1e-6, scale_z=0.0)
    #
    # return output_path

    # Writes a pyramidal, tiled, JPEG-compressed BigTIFF via pyvips - lands back near
    # the original scanner file's size (pylibCZIrw can only write lossless zstd, ~9x
    # bigger) and loads faster in QuPath (precomputed pyramid vs. mosaic re-stitching
    # at read time). Both the flat-field pass and the normalization pass below build a
    # single preallocated numpy array via direct slice-assignment rather than chaining
    # pyvips insert() calls - chained insert() measured ~23x slower at real slide scale
    # and crashed with a stack overflow once the chain got deep enough (excessive
    # recursion in vips's own graph handling), so pyvips is only touched once at the
    # very end, to wrap the finished array and write it. No BGR/RGB flip needed - TIFF
    # has no separate pixel-type tag to fight, unlike CZI, so a true-RGB array just
    # writes as RGB directly. tiffsave's xres/yres are documented in pixels/mm
    # (confirmed via pyvips.Image.tiffsave docstring), unlike pylibCZIrw's
    # write_metadata which silently expected meters despite its docstring.
    czifile = CziFile(path)
    mosaic_bbox = czifile.get_mosaic_bounding_box()
    canvas_w = int(round(mosaic_bbox.w * scale_factor))
    canvas_h = int(round(mosaic_bbox.h * scale_factor))

    # Unscanned corners/edges of the bounding box (mosaic tiles form an irregular
    # blob, not a clean rectangle) never get a tile written and keep whatever the
    # array started as - white matches the scanner's own background convention
    # (and the Beer-Lambert OD math's white-background assumption), unlike black.
    print(f"Flat-field correcting {os.path.basename(path)}...")
    flatfield_array = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    correction_surface = None
    for tile_info, tile_bbox in czifile.get_all_mosaic_tile_bounding_boxes(C=0).items():
        raw_tile = load_raw_tile(czifile, tile_info.m_index)
        ds_tile = downsample_tile(raw_tile, scale_factor)
        if correction_surface is None or correction_surface.shape[:2] != ds_tile.shape[:2]:
            correction_surface = evaluate_flatfield_surface(coeffs_per_channel, ds_tile.shape[0], ds_tile.shape[1])
        corrected_tile = apply_flatfield_correction(ds_tile, correction_surface, target=target)

        x0 = int(round((tile_bbox.x - mosaic_bbox.x) * scale_factor))
        y0 = int(round((tile_bbox.y - mosaic_bbox.y) * scale_factor))
        h, w, _ = corrected_tile.shape
        # Later tiles simply overwrite earlier ones in overlap regions via plain
        # slice-assignment (no blending) - matches read_mosaic's own last-write-wins
        # behavior. Clipped defensively in case independent per-tile rounding ever
        # pushes a tile's far edge past the canvas by a pixel or two.
        y1, x1 = min(y0 + h, canvas_h), min(x0 + w, canvas_w)
        flatfield_array[y0:y1, x0:x1] = corrected_tile[:y1 - y0, :x1 - x0]

    # Preview comes straight from this already-corrected array - no seam/overlap
    # ambiguity (unlike a raw CZI read_mosaic stitch), and no second full-res read.
    preview = downsample_tile(flatfield_array, NORMALIZATION_PREVIEW_SCALE)
    try:
        source_HE, source_maxC = stain_vectors_for_slide(preview)
    except Exception as e:
        print(f"SKIPPING {os.path.basename(path)}: whole-slide stain-vector estimation "
              f"failed ({e}) - no output written for this slide.")
        return None

    normalized_array = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    tile_size = 256
    n_cols = -(-canvas_w // tile_size)  # ceil division
    n_rows = -(-canvas_h // tile_size)
    total_tiles = n_cols * n_rows
    print(f"Normalizing {os.path.basename(path)}: {total_tiles} tiles ({n_cols}x{n_rows} grid)...")
    for y0 in range(0, canvas_h, tile_size):
        for x0 in range(0, canvas_w, tile_size):
            w = min(tile_size, canvas_w - x0)
            h = min(tile_size, canvas_h - y0)
            tile = flatfield_array[y0:y0 + h, x0:x0 + w]
            normalized_array[y0:y0 + h, x0:x0 + w] = normalize_tile_fixed_source(
                tile, source_HE, source_maxC)

    out_pixel_size_um = get_pixel_size_um(czifile) / scale_factor
    pixels_per_mm = 1000.0 / out_pixel_size_um
    canvas = pyvips.Image.new_from_memory(normalized_array.data, canvas_w, canvas_h, 3, "uchar")
    print(f"Saving corrected+normalized output to {output_path}")
    canvas.tiffsave(output_path, tile=True, tile_width=256, tile_height=256, pyramid=True,
                     compression="jpeg", Q=jpeg_quality, bigtiff=True,
                     xres=pixels_per_mm, yres=pixels_per_mm)
    return output_path


def read_corrected_czi(output_path, page=0):
    # czifile = CziFile(output_path)
    # bbox = czifile.get_mosaic_bounding_box()
    # img = czifile.read_mosaic(C=0, scale_factor=scale_factor, region=(bbox.x, bbox.y, bbox.w, bbox.h))[0, :, :, :]
    # return swap_channel_order(img)

    # output_path is now a pyramidal BigTIFF (see write_corrected_czi) - read an
    # embedded pyramid level directly rather than resizing down from full-res page 0:
    # ~135x faster (0.02s vs 2.71s measured on the R23-055 reference) since it decodes
    # only that level's own (much smaller) JPEG tiles instead of every full-res tile.
    # No channel-order fix needed, see write_corrected_czi's comment on why TIFF
    # doesn't have CZI's tagging pitfall.
    return pyvips.Image.tiffload(output_path, page=page).numpy()


#Establish a tiling intensity profile
for rabbit_id, block_no, path in test_slides:
    img, mask, background_tiles, tissue_tiles, dark_anomaly_tiles = find_background_tiles(path)
    if not background_tiles:
        print(f"SKIPPING {os.path.basename(path)} (rabbit {rabbit_id}, block {block_no}): "
              f"no clean background tiles found - cannot fit flat-field correction.")
        continue

    total = len(background_tiles) + len(tissue_tiles) + len(dark_anomaly_tiles)
    print(f"{os.path.basename(path)}: {len(background_tiles)}/{total} tiles clean background, "
          f"{len(dark_anomaly_tiles)} dark anomaly tiles excluded")

    sample_tiles = select_sample_tiles(background_tiles, n=5)
    if SHOW_PLOTS:
        plot_background_tile_overlay(img, background_tiles, highlight_tiles=sample_tiles, title=os.path.basename(path))

    median_map, std_map = build_background_stamp(path, background_tiles)
    fitted_surface, coeffs_per_channel = fit_flatfield_surface(median_map)

    if SHOW_PLOTS:
        plot_tiles_vs_median(path, sample_tiles, median_map, title=os.path.basename(path))
        plot_cross_sections(median_map, fitted_surface, axis="x", n_slices=3,
                             title=os.path.basename(path) + " (horizontal cross-sections)")
        plot_cross_sections(median_map, fitted_surface, axis="y", n_slices=3,
                             title=os.path.basename(path) + " (vertical cross-sections)")
        plot_correction_math_walkthrough(median_map, fitted_surface, axis="y",
                                          title=os.path.basename(path) + " (correction math walkthrough)")
        plot_correction_comparison(path, sample_tiles, fitted_surface,
                                    title=os.path.basename(path) + " (raw vs. corrected background tiles)")

    corrected_slide = reconstruct_corrected_slide(path, coeffs_per_channel, scale_factor=scale_factor)
    if SHOW_PLOTS:
        plt.figure(figsize=(12, 12))
        plt.imshow(corrected_slide)
        plt.title(os.path.basename(path) + " (reconstructed, corrected, tile-by-tile placement)")
        plt.show()

    output_dir = rabbit_block_output_dir(rabbit_id, block_no)
    output_tiff_path = os.path.join(output_dir,
                                     os.path.splitext(os.path.basename(path))[0] + "_corrected_fullres.tif")
    result_path = write_corrected_czi(path, coeffs_per_channel, output_tiff_path, scale_factor=1)
    if result_path is None:
        continue
    if SHOW_PLOTS:
        tiff_roundtrip_img = read_corrected_czi(output_tiff_path, page=4)  # embedded 1/16 level
        plt.figure(figsize=(12, 12))
        plt.imshow(tiff_roundtrip_img)
        plt.title(os.path.basename(path) + " (full-res pyramidal BigTIFF via pyvips, displayed at 1/20)")
        plt.show()
