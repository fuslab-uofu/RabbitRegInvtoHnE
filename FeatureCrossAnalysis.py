from PyQt5.QtWidgets import QApplication
from Viewer import VolumeViewer
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # required for network filesystems (ceph, NFS)
from ApplyTransforms import propagate_tiles_to_space, propagate_tiles_to_Invivo_spaces
from RabbitPathFinder import find_all_the_paths
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import affine_transform, map_coordinates
import pandas as pd
from aicspylibczi import CziFile
import skimage as ski
import json
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import block_reduce
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def ensure_ccw(corners):
    """Reverse corners to CCW if they're CW. corners: (4, 2) array."""
    # Shoelace formula for signed area
    x, y = corners[:, 0], corners[:, 1]
    signed_area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2
    if signed_area < 0:  # clockwise — reverse
        return corners[::-1]
    return corners


def tiles_to_geojson(coords, output_path):
    """
    coords: np.array of shape (N, 4, 2) — N tiles, 4 corners, (x, y)
    """
    features = []
    for i, tile in enumerate(coords):
        corners = ensure_ccw(tile)
        ring = corners[:, [1, 0]].tolist() + [corners[0, [1, 0]].tolist()]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring]
            },
            "properties": {
                "objectType": "annotation",
                "name": f"tile_{i}"
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def mean_nonzero(arr, axis=None, **kwargs):
    masked = np.ma.masked_equal(arr, 0)
    result = np.ma.mean(masked, axis=axis)
    return result.filled(0) if isinstance(result, np.ma.MaskedArray) else result


#Texture features-
def haralick_features(image_gray, ignore_zeros=True):
    # image_gray must be uint8
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True,
                        normed=False)  # normed=False so we can edit first

    if ignore_zeros:
        # Exclude co-occurrences involving background (0) pixels
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0

    # Renormalize
    glcm_sum = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.where(glcm_sum > 0, glcm / glcm_sum, 0)

    # Average over angles
    asm = graycoprops(glcm, 'ASM').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = np.sqrt(asm)  # energy = sqrt(ASM)

    # Remaining 9 computed directly from GLCM
    p = glcm.mean(axis=(2, 3))  # average over distances and angles -> (levels, levels)
    i, j = np.mgrid[0:256, 0:256]
    mu_i = (i * p).sum()
    mu_j = (j * p).sum()
    sig_i = np.sqrt(((i - mu_i) ** 2 * p).sum())
    sig_j = np.sqrt(((j - mu_j) ** 2 * p).sum())

    p_x = p.sum(axis=1)
    p_y = p.sum(axis=0)

        # Simpler direct formulas
    variance = ((i - mu_i) ** 2 * p).sum()
    dissimilarity = (np.abs(i - j) * p).sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_prob = p.max()

    # Sum average, sum variance, sum entropy (use p_xpy = sum along diagonals i+j=k)
    p_sum = np.zeros(2 * 256)
    for k in range(2 * 256):
        mask = (i + j) == k
        p_sum[k] = p[mask].sum()
    sum_avg = np.sum(np.arange(2 * 256) * p_sum)
    sum_entropy = -np.sum(p_sum * np.log(p_sum + 1e-10))
    sum_var = np.sum((np.arange(2 * 256) - sum_entropy) ** 2 * p_sum)

    # Diff entropy and diff variance (p along |i-j|=k diagonals)
    p_diff = np.zeros(256)
    for k in range(256):
        mask = np.abs(i - j) == k
        p_diff[k] = p[mask].sum()
    diff_var = np.sum(np.arange(256) ** 2 * p_diff) - np.sum(np.arange(256) * p_diff) ** 2
    diff_entropy = -np.sum(p_diff * np.log(p_diff + 1e-10))

    return np.array([asm, contrast, correlation, variance, homogeneity,
                     sum_avg, sum_var, sum_entropy, entropy,
                     diff_var, diff_entropy, energy, dissimilarity, max_prob])

def extract_features(czi_patch, tformed_quad, mask, mask_labels):
    """Compute H&E color stats, haralick features, and mask label for one tile.
    Returns a feature dict, or None if the tile should be skipped (ambiguous mask label)."""
    haralick_names = ['asm', 'contrast', 'correlation', 'variance', 'homogeneity',
                      'sum_avg', 'sum_var', 'sum_entropy', 'entropy',
                      'diff_var', 'diff_entropy', 'energy', 'dissimilarity', 'max_prob']

    patch_f = czi_patch.astype(float)
    patch_f[patch_f == 0] = np.nan
    flat = patch_f.reshape(-1, 3)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)

    features = {
        'mean_R': means[0], 'mean_G': means[1], 'mean_B': means[2],
        'std_R': stds[0], 'std_G': stds[1], 'std_B': stds[2],
    }

    redchan = czi_patch[:, :, 0].astype(np.uint8)
    features.update({f'red_{n}': v for n, v in zip(haralick_names, haralick_features(redchan))})
    bluechan = czi_patch[:, :, 2].astype(np.uint8)
    features.update({f'blue_{n}': v for n, v in zip(haralick_names, haralick_features(bluechan))})

    if mask is not None:
        label = label_tile_from_mask(mask, tformed_quad, mask_labels)
        if label is None:
            return None
        features['label'] = label

    #This is not built to go here yet placeholder code will need to be reworked a bit for functionality-
    # ds_list=[]
    # for r in range(5):
    # #Playing with downsampling->
    #     r=r+1
    #     block_size = (50*r, 50*r, 1)
    #     downsampled_array_avg = block_reduce(czi_patch, block_size=block_size, func=mean_nonzero).astype(np.uint8)
    #     ds_list.append(downsampled_array_avg)
    # #Starter Features->>
    #
    # #Mean intensity of all three color channels->
    # nancopy=czi_patch.astype(float)
    # nancopy[nancopy==0]=np.nan
    # HnEFeatures[tile, 2:5]= np.nanmean(np.nanmean(nancopy, axis=0), axis=0)

    # Texture features-
    # 1st convert patch to greyscale->
    # Apply to red and green color channels seperately->
    # redchan= czi_patch[:,:,0].astype(np.uint8)
    # textfeatures_red = haralick_features(redchan)
    # HnEFeatures[tile, 5:19] = textfeatures_red
    #
    # bluechan= czi_patch[:,:,2].astype(np.uint8)
    # textfeatures_blue = haralick_features(bluechan)
    # HnEFeatures[tile, 19:33] = textfeatures_blue

    # for s in range(5):
    #     start= 33+ s*28
    #     redchan = ds_list[s][:, :, 0].astype(np.uint8)
    #     textfeatures_red = haralick_features(redchan)
    #     HnEFeatures[i, start:start+14] = textfeatures_red
    #
    #     bluechan = ds_list[s][:, :, 2].astype(np.uint8)
    #     textfeatures_blue = haralick_features(bluechan)
    #     HnEFeatures[i, start+14:start+28] = textfeatures_blue


    #MR features- Likewise not flushed out at all!
    # dy, dx = np.gradient(MR_Tile)
    #
    # MRFeatures[tile, 2] = np.mean(MR_Tile)
    # MRFeatures[tile, 3] = np.std(MR_Tile)
    # MRFeatures[tile, 4] = np.mean(dx)
    # MRFeatures[tile, 5] = np.mean(dy)

    # print(MRFeatures[tile, :6])

    return features


def load_landmarks(hne_path, img_number):
    """Load the landmarks .npy file containing img_number from the Landmarks folder inside hne_path."""
    landmarks_dir = os.path.join(hne_path, 'Landmarks')
    filepath = next(os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir)
                    if img_number in f and f.endswith('.npy') and not f.startswith('._'))
    return np.load(filepath, allow_pickle=True)

def get_bf_slice_index(bf_cropped_dir, img_number):
    """Return the NIfTI slice index of the blockface image matching img_number.
    All tiffs (including _scatter_fill) are included in ordering to match NIfTI build order,
    but the match only targets _scatter.tiff files."""
    all_files = sorted(f for f in os.listdir(bf_cropped_dir)
                       if f.endswith('.tiff') and not f.startswith('._'))
    match = next(f for f in all_files if img_number in f and f.endswith('_scatter.tiff'))
    return all_files.index(match)

def CSZ_CZI_lookup(rab_ID, block, file_numb):
    BaseCephPath = "/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/"
    block_no = block.lower()
    print(block_no)
    # Path to CSV-
    csv_dirpath = os.path.join(BaseCephPath, rab_ID, rab_ID + "_BlockFaceImages", block_no, "csv_files")
    files = [f for f in os.listdir(csv_dirpath) if f.endswith('csv')]
    sorted_filenames = sorted(files, key=lambda x: float(re.findall(r'\d+', x)[0]))
    csv_file = sorted_filenames[-1]
    csv_filepath = os.path.join(csv_dirpath, csv_file)
    # Load in the csv file to get the proper term to rename czi files-
    df = pd.read_csv(csv_filepath, header=None)
    row = df[df.iloc[:, 0].str.contains(file_numb)]
    czi_name = row.iloc[0, 2].split(';')[0]
    # Path to CZI directory-
    czi_dirpath = os.path.join(BaseCephPath, rab_ID, rab_ID + "_HnE_5x", block_no)
    czi_file = next(f for f in os.listdir(czi_dirpath) if czi_name in f)
    czi_path = os.path.join(czi_dirpath, czi_file)
    return czi_path


def draw_line_3d(vol, p0, p1, val):
    """Rasterize a 3D line segment from p0 to p1 into vol by linear interpolation."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    n = int(np.ceil(np.linalg.norm(p1 - p0))) + 1
    for t in np.linspace(0, 1, n):
        pt = np.clip(np.round(p0 + t * (p1 - p0)).astype(int), 0, np.array(vol.shape) - 1)
        vol[pt[0], pt[1], pt[2]] = val


def norm255(arr):
    m = np.max(arr)
    return arr / m * 255 if m > 0 else arr

def tiling_tool(twoDIm, tile_size):
    #Mask out background->
    rgbmean = np.mean(twoDIm, axis=2)
    whiteIm=np.where(rgbmean>210,0,1)
    twoDIm= twoDIm * whiteIm[:, :, np.newaxis]
    print(twoDIm.shape[0], twoDIm.shape[1])
    yrem= twoDIm.shape[0]%tile_size
    if yrem:
        ystart= yrem//2
        ycount=int(twoDIm.shape[0]//tile_size)
    else:
        ycount=int(twoDIm.shape[0]/tile_size)
        ystart=0
    xrem= twoDIm.shape[1]%tile_size
    if xrem:
        xstart= xrem//2
        xcount=int(twoDIm.shape[1]//tile_size)
    else:
        xcount=int(twoDIm.shape[1]/tile_size)
        xstart=0

    print(ystart, xstart, ycount, xcount)

    tilesList=[]
    for row in range(ycount):
        for col in range(xcount):
            tile = twoDIm[ystart+row*tile_size:ystart+(row+1)*tile_size, xstart+col*tile_size:xstart+(col+1)*tile_size, :]
            zero_mask = np.all(tile == 0, axis=-1)  # shape: (H, W) boolean
            zero_count = np.sum(zero_mask)
            if zero_count < (tile_size**2)/6:
                org= [ystart+row*tile_size, xstart+col*tile_size]
                tilesList.append(org)

    tilesarray=np.asarray(tilesList)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(twoDIm)  # drop cmap if RGB-

    for tile in range(tilesarray.shape[0]):
        row= tilesarray[tile,0]
        col= tilesarray[tile,1]
        color = 'green'
        rect = patches.Rectangle(
            (col, row),  # note: matplotlib uses (x, y) = (col, row)
            tile_size, tile_size,
            linewidth=1, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()
    return tilesarray


def label_tile_from_mask(mask, tformed_quad, mask_labels, mask_ds=20, threshold=0.8):
    """
    Assign a tissue class to a tile by sampling the annotation mask.
    Returns the class label string, or None if the tile is ambiguous (no single
    class covers >= threshold fraction of pixels).

    tformed_quad: (4, 2) array of tile corners in full-resolution CZI pixel space [row, col]
    mask: 2D uint8 array in CZI/mask_ds pixel space
    mask_labels: dict mapping mask integer values to class name strings
    """
    r_min, c_min = (tformed_quad.min(axis=0) / mask_ds).astype(int)
    r_max, c_max = (tformed_quad.max(axis=0) / mask_ds).astype(int)

    r_min = max(0, r_min)
    c_min = max(0, c_min)
    r_max = min(mask.shape[0], r_max)
    c_max = min(mask.shape[1], c_max)

    tile_region = mask[r_min:r_max, c_min:c_max]
    if tile_region.size == 0:
        return None

    values, counts = np.unique(tile_region, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_frac = counts[dominant_idx] / tile_region.size

    if dominant_frac < threshold:
        return None

    return mask_labels.get(int(values[dominant_idx]), None)


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID     = "R23-055"
    Block        = 7
    target_space = "InVivo"

    hne_base_dir    = os.path.join(RabbitFolder, RabbitID, 'HnE', f'Block{Block:02d}')
    reg_HnE_dir     = os.path.join(hne_base_dir, 'Registered')
    BlockFaceFolder = os.path.join(RabbitFolder, RabbitID, 'BlockFace_RGB', f'Block{Block:02d}')
    #reg_show_path    = os.path.join(BlockFaceFolder, 'greyscale_downsampled.nii.gz')
    reg_show_path = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToBlock07_0427-1855.nii.gz"
    reg_show_path2 = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToBlock07_0428-1042.nii.gz"
    #Need- to determine proper slice in 3d volume-
    bf_cropped_dir  = os.path.join(BlockFaceFolder, 'CroppedImages')
    output_dir = '/Users/jbonaventura/Desktop/Annotations'

    # --- Annotation config ---
    mask_path = None  # set to TIFF output of WorkingGeoJson.py, or None to skip annotation labelling
    #mask_path = '/Users/jbonaventura/Desktop/Annotations/HnE_R23-055_H7_7a_annotations_Mask.tiff'
    if mask_path is not None:
        # mask_labels values must match tisslabels dict in WorkingGeoJson.py; 0 = unannotated (Muscle)
        mask_labels = {0: 'Muscle', 100: 'Necrotic Tissue', 200: 'Immune Infiltration'}
        mask = np.array(Image.open(mask_path))
    else:
        mask = None

    show_tiles = True         # set True to visualize each tile during extraction
    save_tile_overlays = False # set True to save tile-marked NIfTI copies for each Day3/Day0 vol
    extract_features = False  # set True to compute H&E/MR features and save CSV
    Extend_Thru_InVivo = True # propagate Tiles to other InVivo Spaces

    #Load in volume to show tiles on/from-
    reg_show = nib.load(reg_show_path)
    reg_show_affine = reg_show.affine
    reg_show_arr = reg_show.get_fdata()
    #If canonical move over to blockface space for visualization->
    if reg_show.affine[0, 0] > 0 and reg_show.affine[1, 1] > 0:
        reg_show_arr = reg_show_arr[::-1, ::-1, :]

    #Real quick to compare iterp modes-    #Load in volume to show tiles on/from-
    reg_show2 = nib.load(reg_show_path2)
    reg_show2_affine = reg_show2.affine
    reg_show2_arr = reg_show2.get_fdata()
    #If canonical move over to blockface space for visualization->
    if reg_show2.affine[0, 0] > 0 and reg_show2.affine[1, 1] > 0:
        reg_show2_arr = reg_show2_arr[::-1, ::-1, :]


    _target_nib = nib.load(str(find_all_the_paths(RabbitID, Block, RabbitFolder, target_space)['Moving_FilePath']))
    target_arr  = _target_nib.get_fdata()

    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir) if f.endswith('Reg.png') and not f.startswith('._'))
    hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in hne_filenames]
    reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)

    tilesize = 300
    for img in range(1):
        img=5
        hne_ds_im= reg_HnE_arr[:,:,img,:]

        img_number = re.search(r'\d+', hne_filenames[img]).group()  # e.g. '0011'
        landmarks = load_landmarks(hne_base_dir, img_number)
        scale_fac=20 #scale factor between HnE raw and downsampled
        src = np.array([[p.y(), p.x()] for p in landmarks[0]])*3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]])*3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)
        splines_inv = ski.transform.ThinPlateSplineTransform.from_estimate(dst, src)

        CZI_filepath = CSZ_CZI_lookup(RabbitID, f'Block{Block:02d}', img_number)
        slide_id = os.path.splitext(os.path.basename(CZI_filepath))[0]
        print(slide_id)
        output_csv_path = os.path.join(output_dir, f'{slide_id}_features.csv')  #New csv and geojson files for each HnE slide
        output_geojson_path = os.path.join(output_dir, f'{slide_id}_tiles.geojson')

        #Use blockface filename position in CroppedImages to find the correct NIfTI slice-
        slice_num = get_bf_slice_index(bf_cropped_dir, img_number)
        #BlockFace NIfTI has transposed X/Y — apply .T to correct orientation-
        Show_Slice = reg_show_arr[:,:,slice_num].T
        print("Show_Slice info", np.max(Show_Slice), np.min(Show_Slice))
        #Upsampling by 4 this is for visualization and tile extraction purposes- not for analysis-
        vol_upsampled = np.repeat(np.repeat(Show_Slice, 4, axis=0), 4, axis=1)
        Show_Slice_us= vol_upsampled[:hne_ds_im.shape[0],:hne_ds_im.shape[1]]

        #rep to comp interps-
        Show_Slice2 = reg_show2_arr[:,:,slice_num].T
        #Upsampling by 4 this is for visualization and tile extraction purposes- not for analysis-
        vol_upsampled2 = np.repeat(np.repeat(Show_Slice2, 4, axis=0), 4, axis=1)
        Show_Slice_us2= vol_upsampled2[:hne_ds_im.shape[0],:hne_ds_im.shape[1]]



        origin_list = tiling_tool(reg_HnE_arr[:,:,img,:], tilesize)
        czifile = CziFile(CZI_filepath)
        bbox = czifile.get_mosaic_bounding_box()
        # czi_patch = czifile.read_mosaic(C=0, scale_factor=1/10, region=(bbox.x, bbox.y, bbox.w, bbox.h))[0]
        # czi_patch[:, :, [0, 2]] = czi_patch[:, :, [2, 0]]  #Swap color channels for RGB vs BGR conventions

        #output_image = ski.transform.warp(hne_ds_im, splines_inv, output_shape=(czi_img.shape[0]*scale_fac, czi_img.shape[1]*scale_fac, czi_img.shape[2]))
        ## normalize and convert to uint8
        #output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)
        # tformed_origins = splines(origin_list[:, ::-1])[:, ::-1]* 2


        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(hne_ds_im)
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle(
                (col, row),  # note: matplotlib uses (x, y) = (col, row)
                tilesize, tilesize,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)

        axes[1].imshow(Show_Slice_us, cmap='gray')
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle(
                (col, row),  # note: matplotlib uses (x, y) = (col, row)
                tilesize, tilesize,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            axes[1].add_patch(rect)
        plt.show()

        # Build (N_tiles, 4, 2) corner array in [row, col] and propagate all tiles at once-
        rows = origin_list[:, 0]
        cols = origin_list[:, 1]
        all_corners = np.stack([
            np.column_stack([rows,            cols           ]),
            np.column_stack([rows + tilesize, cols           ]),
            np.column_stack([rows + tilesize, cols + tilesize]),
            np.column_stack([rows,            cols + tilesize]),
        ], axis=1)  # (N_tiles, 4, 2)

        # Transform all tile corners backwards to specified target_space.
        target_corners = propagate_tiles_to_space(
            all_corners, slice_num, RabbitID, Block, RabbitFolder, target_space
        )  # (N_tiles, 4, 3) raw voxel coords in target_space


        if target_space == "InVivo" and Extend_Thru_InVivo:
            day3_tile_corners = propagate_tiles_to_Invivo_spaces(target_corners, RabbitID, RabbitFolder, Block)
            if show_tiles or save_tile_overlays:
                from RabbitPathFinder import find_day3_paths, find_day0_paths
                from ApplyTransforms import propagate_tiles_to_day0
                _d3 = find_day3_paths(RabbitID, RabbitFolder)
                day3_arrs = {}
                day3_affines = {}
                day3_headers = {}
                for p in _d3['start_vols'] + _d3['fixed_vols']:
                    stem = os.path.basename(p).replace('.nii.gz', '')
                    _nib = nib.load(p)
                    day3_arrs[stem]    = _nib.get_fdata()
                    day3_affines[stem] = _nib.affine
                    day3_headers[stem] = _nib.header

                structural_stem = os.path.basename(_d3['fixed_vols'][0]).replace('.nii.gz', '')
                _tf = _d3['aps_transform_folder']
                _affine_field  = np.load(os.path.join(_tf, 'Affine_deformation.npy'))
                _splines_field = np.load(os.path.join(_tf, 'SplinesProjection.npy'))
                _sdiff_field   = np.load(os.path.join(_tf, 'sdiff.npy'))
                day0_tile_corners = propagate_tiles_to_day0(
                    day3_tile_corners[structural_stem],
                    _affine_field, _splines_field, _sdiff_field
                )

                _d0 = find_day0_paths(RabbitID, RabbitFolder)
                day0_arrs = {}
                day0_affines = {}
                day0_headers = {}
                for p in _d0['vols']:
                    stem = os.path.basename(p).replace('.nii.gz', '')
                    _nib = nib.load(p)
                    day0_arrs[stem]    = _nib.get_fdata()
                    day0_affines[stem] = _nib.affine
                    day0_headers[stem] = _nib.header

                if show_tiles:
                    day3_patch_vals = {stem: [] for stem in day3_arrs}
                    day0_patch_vals = {stem: [] for stem in day0_arrs}

                if save_tile_overlays:
                    day3_marked_vols  = {stem: arr.copy() for stem, arr in day3_arrs.items()}
                    day3_marker_vals  = {stem: float(arr.max()) for stem, arr in day3_arrs.items()}
                    day0_marked_vols  = {stem: arr.copy() for stem, arr in day0_arrs.items()}
                    day0_marker_vals  = {stem: float(arr.max()) for stem, arr in day0_arrs.items()}

        # Transform all tile corners to full-res HnE space-
        all_tformed_quads = (splines(all_corners.reshape(-1, 2))[:, ::-1] * scale_fac).reshape(len(origin_list), 4, 2)  # (N_tiles, 4, 2) in [col, row]

        tile_marked_vol = target_arr.copy()
        marker_val = float(target_arr.max())

        #Tile-Wise work through-
        records = []

        MRFeatures= np.zeros((len(origin_list), 19))
        MRFeatures[:,:2]=origin_list

        transformed_originList=[]
        tile_render_data = []
        for tile in range(len(origin_list)):
        #for tile in range(3):
            O_up = origin_list[tile]
            tformed_quad = all_tformed_quads[tile]
            transformed_originList.append(tformed_quad)
            # Bounding box in CZI space — clamped to image bounds
            r_min, c_min = tformed_quad.min(axis=0).astype(int)
            r_max, c_max = tformed_quad.max(axis=0).astype(int)

            x = max(bbox.x, bbox.x + c_min)
            y = max(bbox.y, bbox.y + r_min)
            x_end = min(bbox.x + bbox.w, bbox.x + c_max)
            y_end = min(bbox.y + bbox.h, bbox.y + r_max)
            w = x_end - x
            h = y_end - y

            if w <= 0 or h <= 0:
                continue  # tile completely outside image bounds

            region = (x, y, w, h)

            czi_patch = czifile.read_mosaic(C=0, scale_factor=1, region=region)[0]
            #Swap red and blue to be consistent with other file-
            czi_patch[:, :, [0, 2]] = czi_patch[:, :, [2, 0]]

            # --- Feature extraction ---
            czi_r_min, czi_c_min = tformed_quad.min(axis=0).astype(int)
            czi_r_max, czi_c_max = tformed_quad.max(axis=0).astype(int)
            record = {'tile_id': tile, 'origin_row': O_up[0], 'origin_col': O_up[1],
                      'czi_r_min': czi_r_min, 'czi_c_min': czi_c_min,
                      'czi_r_max': czi_r_max, 'czi_c_max': czi_c_max}

            if extract_features:
                hne_features = extract_features(czi_patch, tformed_quad, mask, mask_labels)
                if hne_features is None:
                    continue
                record.update(hne_features)

            # Target space voxel coords for this tile
            tile_vox = target_corners[tile]  # (4, 3)
            print("tile_vox", tile_vox)
            t_z   = int(np.clip(np.round(tile_vox[:, 2].mean()), 0, target_arr.shape[2] - 1))
            t_x0  = max(0, int(tile_vox[:, 0].min()))
            t_x1  = min(target_arr.shape[0], int(tile_vox[:, 0].max()) + 1)
            t_y0  = max(0, int(tile_vox[:, 1].min()))
            t_y1  = min(target_arr.shape[1], int(tile_vox[:, 1].max()) + 1)
            for k in range(4):
                draw_line_3d(tile_marked_vol, tile_vox[k], tile_vox[(k + 1) % 4], marker_val)

            if extract_features:
                record.update({f'{target_space}_z': t_z,
                               f'{target_space}_x0': t_x0, f'{target_space}_x1': t_x1,
                               f'{target_space}_y0': t_y0, f'{target_space}_y1': t_y1})
                records.append(record)

            #Masking to only get polygon->
            verticies = tformed_quad[:,::-1].copy()
            verticies -= [c_min, r_min]
            verticies = verticies.astype(np.int32)

            #Mask out bright background->
            tilewarp = np.zeros(czi_patch.shape[:2], dtype=np.uint8)
            cv2.fillPoly(tilewarp, [verticies], 255)
            czi_patch = czi_patch * (tilewarp[:, :, np.newaxis] > 0)
            #Mask out bright background->
            rgbmean = np.mean(czi_patch, axis=2)
            whiteIm = np.where(rgbmean > 210, 0, 1)
            czi_patch = czi_patch * whiteIm[:, :, np.newaxis]


            # #Okay, now we want to extend this to the MR data!
            # #pull cooresponding tile from MRSlice->
            MR_Tile = Show_Slice_us[O_up[0]:O_up[0]+tilesize, O_up[1]:O_up[1]+tilesize]

            MR_Tile2 = Show_Slice_us2[O_up[0]:O_up[0] + tilesize, O_up[1]:O_up[1] + tilesize]


            # Original approach: axis-aligned bounding box at mean z
            target_patch = target_arr[t_x0:t_x1, t_y0:t_y1, t_z]

            # Oblique reslice: bilinear interpolation across all 4 corners
            res1 = max(1, int(np.linalg.norm(tile_vox[1] - tile_vox[0])))
            res2 = max(1, int(np.linalg.norm(tile_vox[3] - tile_vox[0])))
            print("reses-", res1, res2)
            ss, tt = np.meshgrid(np.linspace(0, 1, res1), np.linspace(0, 1, res2))
            points = (
                (1 - ss)[..., None] * (1 - tt)[..., None] * tile_vox[0] +
                     ss[..., None]  * (1 - tt)[..., None] * tile_vox[1] +
                     ss[..., None]  *      tt[..., None]  * tile_vox[2] +
                (1 - ss)[..., None] *      tt[..., None]  * tile_vox[3]
            )  # (res2, res1, 3)
            print(points.shape)
            coords = points.reshape(-1, 3).T  # (3, N)
            print(coords.shape)
            target_patch_oblique = map_coordinates(target_arr, coords, order=0).reshape(res2, res1)
            print(target_patch_oblique.shape)

            vmin = np.percentile(target_arr, 2)
            vmax = np.percentile(target_arr, 98)

            if show_tiles:
                d3_patches = {}
                d0_patches = {}
                if target_space == "InVivo" and Extend_Thru_InVivo:
                    for stem, arr in day3_arrs.items():
                        d3_vox = day3_tile_corners[stem][tile]  # (4, 3)
                        r1 = max(1, int(np.linalg.norm(d3_vox[1] - d3_vox[0])))
                        r2 = max(1, int(np.linalg.norm(d3_vox[3] - d3_vox[0])))
                        ss_d, tt_d = np.meshgrid(np.linspace(0, 1, r1), np.linspace(0, 1, r2))
                        pts_d = (
                            (1 - ss_d)[..., None] * (1 - tt_d)[..., None] * d3_vox[0] +
                                 ss_d[..., None]  * (1 - tt_d)[..., None] * d3_vox[1] +
                                 ss_d[..., None]  *      tt_d[..., None]  * d3_vox[2] +
                            (1 - ss_d)[..., None] *      tt_d[..., None]  * d3_vox[3]
                        )
                        patch = map_coordinates(arr, pts_d.reshape(-1, 3).T, order=0).reshape(r2, r1)
                        d3_patches[stem] = patch
                        day3_patch_vals[stem].append(patch.flatten())

                    # All Day0 volumes share the same grid, so compute the reslice coords once
                    d0_vox = day0_tile_corners[tile]  # (4, 3)
                    r1_d0 = max(1, int(np.linalg.norm(d0_vox[1] - d0_vox[0])))
                    r2_d0 = max(1, int(np.linalg.norm(d0_vox[3] - d0_vox[0])))
                    ss_d0, tt_d0 = np.meshgrid(np.linspace(0, 1, r1_d0), np.linspace(0, 1, r2_d0))
                    pts_d0 = (
                        (1 - ss_d0)[..., None] * (1 - tt_d0)[..., None] * d0_vox[0] +
                             ss_d0[..., None]  * (1 - tt_d0)[..., None] * d0_vox[1] +
                             ss_d0[..., None]  *      tt_d0[..., None]  * d0_vox[2] +
                        (1 - ss_d0)[..., None] *      tt_d0[..., None]  * d0_vox[3]
                    )
                    coords_d0 = pts_d0.reshape(-1, 3).T
                    for stem, arr in day0_arrs.items():
                        patch = map_coordinates(arr, coords_d0, order=0).reshape(r2_d0, r1_d0)
                        d0_patches[stem] = patch
                        day0_patch_vals[stem].append(patch.flatten())

                tile_render_data.append({
                    'czi_patch': czi_patch,
                    'MR_Tile2': MR_Tile2,
                    'target_patch_oblique': target_patch_oblique,
                    'vmin': vmin,
                    'vmax': vmax,
                    'd3_patches': d3_patches,
                    'd0_patches': d0_patches,
                })

            if save_tile_overlays and target_space == "InVivo" and Extend_Thru_InVivo:
                for stem in day3_marked_vols:
                    d3_vox = day3_tile_corners[stem][tile]
                    for k in range(4):
                        draw_line_3d(day3_marked_vols[stem], d3_vox[k], d3_vox[(k + 1) % 4], day3_marker_vals[stem])
                d0_vox = day0_tile_corners[tile]
                for stem in day0_marked_vols:
                    for k in range(4):
                        draw_line_3d(day0_marked_vols[stem], d0_vox[k], d0_vox[(k + 1) % 4], day0_marker_vals[stem])

            #
            # fig, axes = plt.subplots(1, 7, figsize=(20, 4))
            # axes[0].imshow(np.fliplr(np.rot90(czi_patch, k=2)))
            # axes[0].set_title('Original H&E')
            # for q in range(5):
            #     axes[q+1].imshow(np.fliplr(np.rot90(ds_list[q], k=2)))
            #     axes[q+1].set_title('H&E Downsampled by ' + str(50*(q+1)))
            # axes[6].imshow(MR_Tile)
            # axes[6].set_title("MR Tile")
            # plt.tight_layout()
            # plt.show()

            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(hne_ds_im)
            # axes[0].scatter(quadlist[:,1], quadlist[:,0], c='r', marker='o', s=5)
            # axes[1].imshow(czi_patch)
            # #axes[1].scatter(verticies[:,0], verticies[:,1], c='r', marker='o', s=5)
            # axes[2].imshow(MR_Tile)
            # plt.show()

        if show_tiles:
            day3_clim = {}
            day0_clim = {}
            if target_space == "InVivo" and Extend_Thru_InVivo:
                for stem, vals_list in day3_patch_vals.items():
                    vals = np.concatenate(vals_list)
                    vals = vals[vals > 0]
                    if len(vals) > 0:
                        day3_clim[stem] = (np.percentile(vals, 5), np.percentile(vals, 98))
                for stem, vals_list in day0_patch_vals.items():
                    vals = np.concatenate(vals_list)
                    vals = vals[vals > 0]
                    if len(vals) > 0:
                        day0_clim[stem] = (np.percentile(vals, 5), np.percentile(vals, 98))

            for rd in tile_render_data:
                n_d3 = len(rd['d3_patches'])
                n_d0 = len(rd['d0_patches'])
                n_extra = n_d3 + n_d0
                fig, axes = plt.subplots(1, 3 + n_extra, figsize=(4 * (4 + n_extra), 4))
                axes[0].imshow(np.rot90(rd['czi_patch'], k=3))
                axes[0].set_title('Original H&E')
                axes[1].imshow(rd['MR_Tile2'], cmap='gray', vmin=rd['vmin'], vmax=rd['vmax'])
                axes[1].set_title('T1w CE MR')
                axes[2].imshow(np.rot90(np.fliplr(rd['target_patch_oblique'])), cmap='gray', vmin=rd['vmin'], vmax=rd['vmax'])
                axes[2].set_title('Tile Prop Back to In.V.')
                for i, (stem, patch) in enumerate(rd['d3_patches'].items()):
                    vmin_d3, vmax_d3 = day3_clim[stem]
                    axes[3 + i].imshow(np.rot90(np.fliplr(patch)), cmap='gray', vmin=vmin_d3, vmax=vmax_d3)
                    axes[3 + i].set_title('D3 ' + stem[:8])
                for i, (stem, patch) in enumerate(rd['d0_patches'].items()):
                    vmin_d0, vmax_d0 = day0_clim[stem]
                    axes[3 + n_d3 + i].imshow(np.rot90(np.fliplr(patch)), cmap='gray', vmin=vmin_d0, vmax=vmax_d0)
                    axes[3 + n_d3 + i].set_title('D0 ' + stem[:8])
                plt.tight_layout()
                plt.show()

        if save_tile_overlays and target_space == "InVivo" and Extend_Thru_InVivo:
            for stem, marked_vol in day3_marked_vols.items():
                save_path = os.path.join(output_dir, f'{slide_id}_Day3_{stem}_tile_overlay.nii.gz')
                nib.save(nib.Nifti1Image(marked_vol.astype(np.float32), day3_affines[stem], day3_headers[stem]), save_path)
                print(f'Saved Day3 tile overlay to {save_path}')
            for stem, marked_vol in day0_marked_vols.items():
                save_path = os.path.join(output_dir, f'{slide_id}_Day0_{stem}_tile_overlay.nii.gz')
                nib.save(nib.Nifti1Image(marked_vol.astype(np.float32), day0_affines[stem], day0_headers[stem]), save_path)
                print(f'Saved Day0 tile overlay to {save_path}')

        # print(transformed_originList)
        originarray= np.array(transformed_originList)
        tiles_to_geojson(originarray, output_geojson_path)

        tile_nifti_path = os.path.join(output_dir, f'{slide_id}_{target_space}_tiles2.nii.gz')
        nib.save(nib.Nifti1Image(tile_marked_vol.astype(np.float32), _target_nib.affine, _target_nib.header), tile_nifti_path)
        print(f'Saved tile-marked {target_space} volume to {tile_nifti_path}')

        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_csv_path, index=False)
            print(f'Saved {len(df)} labeled tiles to {output_csv_path}')
    #for plotting-
    # HneXs= HnEFeatures[:,1]
    # HneYs= HnEFeatures[:,0]
    # HneVals= HnEFeatures[:,2]
    # MRXs= MRFeatures[:,1]
    # MRYs= MRFeatures[:,0]
    # MRVals= MRFeatures[:,2]
    #
    #
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].scatter(MRXs, MRYs, c=MRVals, marker='o', s=20)
    # axes[0].margins(0.1)
    # axes[0].set_title('MR values')
    # axes[1].scatter(HneXs, HneYs, c=HneVals, marker='o', s=20)
    # axes[1].margins(0.1)
    # axes[1].set_title('HnE values')
    # axes[0].set_aspect('equal')
    # axes[1].set_aspect('equal')
    # plt.show()
    #
    # MRstart=2
    # MRend=6
    # lenMRfeats= MRend-MRstart
    # corr = np.corrcoef(MRFeatures[:,MRstart:MRend].T, HnEFeatures[:,2:].T)[:lenMRfeats, lenMRfeats:]
    # fig, ax = plt.subplots()
    # im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.colorbar(im)
    # plt.show()



    # app = QApplication(sys.argv)
    # viewer_reg = VolumeViewer(norm255(reg_show_resampled),  reg_HnE_arr,
    #                           'Registered InVivo — HnE')
    # viewer_reg.show()
    # sys.exit(app.exec_())
