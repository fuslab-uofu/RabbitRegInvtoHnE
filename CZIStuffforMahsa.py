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
from RabbitPathFinder import find_day3_paths, find_day0_paths
from ApplyTransforms import propagate_tiles_to_day0
from TileUtils import tiling_tool, load_landmarks, get_bf_slice_index, CSV_CZI_lookup
from HnEFeatureExtraction import mean_nonzero, haralick_features, extract_tile_features


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


def tile_class_composition(mask, tformed_quad, mask_labels, mask_ds=20):
    """
    Compute the fraction of each annotated class within a tile's footprint
    in the annotation mask. Returns a dict mapping class name -> fraction
    (fractions sum to ~1), or None if the tile falls entirely outside the mask.

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
    total = tile_region.size
    return {mask_labels.get(int(v), f'unknown_{int(v)}'): c / total for v, c in zip(values, counts)}


if __name__ == '__main__':
    RabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    RabbitID     = "R23-055"
    Block        = 7
    target_space = "InVivo"

    hne_base_dir    = os.path.join(RabbitFolder, RabbitID, 'HnE', f'Block{Block:02d}')
    reg_HnE_dir     = os.path.join(hne_base_dir, 'Registered')
    BlockFaceFolder = os.path.join(RabbitFolder, RabbitID, 'BlockFace_RGB', f'Block{Block:02d}')
    #Need- to determine proper slice in 3d volume-
    bf_cropped_dir  = os.path.join(BlockFaceFolder, 'CroppedImages')
    #Change to whatevers convenient to you-
    output_dir = '/Users/jbonaventura/Desktop/Annotations'

    # --- Annotation config ---
    #mask_path = None  # set to TIFF output of WorkingGeoJson.py, or None to skip annotation labeling
    mask_path = '/Users/jbonaventura/Desktop/Annotations/HnE_R23-055_H7_7a_annotations_Mask.tiff'
    if mask_path is not None:
        # mask_labels values must match tisslabels dict in WorkingGeoJson.py; 0 = unannotated (Muscle)
        mask_labels = {0: 'Muscle', 100: 'Necrotic Tissue', 200: 'Immune Infiltration'}
        mask = np.array(Image.open(mask_path))
    else:
        mask = None

    show_tiles = True         # set True to visualize each tile during extraction
    extract_features = False  # set True to compute H&E/MR features and save CSV

    ##May need to add handling here not sure how consistent we've been with our Reg names->
    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir) if f.endswith('Reg.png') and not f.startswith('._'))
    hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in hne_filenames]
    reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)

    tilesize = 300
    for img in range(1):
        #To just work with one at a time->
        img=5
        hne_ds_im= reg_HnE_arr[:,:,img,:]

        #Finding tps transform between CZI and blockface->
        img_number = re.search(r'\d+', hne_filenames[img]).group()  # e.g. '0011'
        landmarks = load_landmarks(hne_base_dir, img_number)
        scale_fac=20 #scale factor between HnE raw and downsampled from CZItoTIFF
        #An annoying factor of three is applied because that was used for drawing landmarks-
        src = np.array([[p.x(), p.y()] for p in landmarks[0]])*3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]])*3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)
        splines_inv = ski.transform.ThinPlateSplineTransform.from_estimate(dst, src)

        #Get path to actual CZI filepath->
        CZI_filepath = CSV_CZI_lookup(RabbitID, f'Block{Block:02d}', img_number)
        slide_id = os.path.splitext(os.path.basename(CZI_filepath))[0]
        print(slide_id)
        #Some autopath generation for saving features and tiles if you want that->
        output_csv_path = os.path.join(output_dir, f'{slide_id}_features.csv')  #New csv and geojson files for each HnE slide
        output_geojson_path = os.path.join(output_dir, f'{slide_id}_tiles.geojson')
        output_tile_labels_csv_path = os.path.join(output_dir, f'{slide_id}_tile_labels.csv')

        #Use blockface filename position in CroppedImages to find the correct NIfTI slice-
        #May not be applicable yet but will be used to relate back to MR data->
        slice_num = get_bf_slice_index(bf_cropped_dir, img_number)

        origin_list = tiling_tool(reg_HnE_arr[:,:,img,:], tilesize)
        czifile = CziFile(CZI_filepath)
        bbox = czifile.get_mosaic_bounding_box()

        print("bboxes",bbox.x, bbox.y)
        # # If we want to look at a downsampled CZI file- useful for verifying splines transform is acting how we want-
        # newscale=1/10
        # sf=newscale/(1/20)
        # print("sf",sf)
        # czi_img = czifile.read_mosaic(C=0, scale_factor=newscale, region=(bbox.x, bbox.y, bbox.w, bbox.h))[0]
        # czi_img[:, :, [0, 2]] = czi_img[:, :, [2, 0]]  #Swap color channels for RGB vs BGR conventions
        # plt.imshow(czi_img)
        # plt.show()
        #
        # output_image1 = ski.transform.warp(hne_ds_im, splines_inv, output_shape=(czi_img.shape[0]/sf, czi_img.shape[1]/sf, czi_img.shape[2]))
        # output_image1 = (output_image1 / np.max(output_image1) * 255).astype(np.uint8)
        #
        # #Need to rescale splines for different sized CZI's->
        # srcrs = np.array([[p.x(), p.y()] for p in landmarks[0]])*3*sf  # fixed
        # dstrs = np.array([[p.x(), p.y()] for p in landmarks[1]])*3*sf  # H&E
        # splinesrs = ski.transform.ThinPlateSplineTransform.from_estimate(srcrs, dstrs)
        #
        # output_image2 = ski.transform.warp(czi_img, splinesrs, output_shape=(hne_ds_im.shape[0]*sf, hne_ds_im.shape[1]*sf,hne_ds_im.shape[2]))
        # # normalize and convert to uint8
        # output_image2 = (output_image2 / np.max(output_image2) * 255).astype(np.uint8)
        #
        # fig, axes = plt.subplots(2, 2)
        # axes[0,0].imshow(hne_ds_im)
        # axes[0,0].set_title('Hne BF Reg Image')
        # axes[0,1].imshow(output_image1)
        # axes[0,1].set_title('BF Reg Image inv splines to ds and unreg hne')
        # axes[1,0].imshow(czi_img)
        # axes[1,0].set_title('Hne CZI from File')
        # axes[1,1].imshow(output_image2)
        # axes[1,1].set_title('HnE splines to bf')
        # plt.tight_layout()
        # plt.show()
        #
        # #Showing Tiles over slices in blockface space->
        # fig, axes = plt.subplots(1, 1)
        # axes.imshow(hne_ds_im)
        # for q in range(len(origin_list)):
        #     row = origin_list[q,0]
        #     col = origin_list[q,1]
        #     color = 'black'
        #     rect = patches.Rectangle(
        #         (col, row),  # note: matplotlib uses (x, y) = (col, row)
        #         tilesize, tilesize,
        #         linewidth=1, edgecolor=color, facecolor='none'
        #     )
        #     axes.add_patch(rect)
        # plt.show()

        # Build (N_tiles, 4, 2) corner array in [row, col] and propagate all tiles at once to other spaces-
        rows = origin_list[:, 0]
        cols = origin_list[:, 1]
        all_corners = np.stack([
            np.column_stack([rows,            cols           ]),
            np.column_stack([rows + tilesize, cols           ]),
            np.column_stack([rows + tilesize, cols + tilesize]),
            np.column_stack([rows,            cols + tilesize]),
        ], axis=1)  # (N_tiles, 4, 2)

        # Transform all tile corners to full-res HnE space-
        #all_tformed_quads = (splines(all_corners.reshape(-1, 2))[:, ::-1] * scale_fac).reshape(len(origin_list), 4, 2)  # (N_tiles, 4, 2) in [col, row]
        # line 225 — flip input to match (x,y) convention, flip output back to (row,col)
        all_tformed_quads = (splines(all_corners.reshape(-1, 2)[:, ::-1])[:, ::-1] * scale_fac).reshape(len(origin_list), 4, 2)

        #Tile-Wise work through-
        transformed_originList=[]
        tile_render_data = []
        records = []
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
                hne_features = extract_tile_features(czi_patch, tformed_quad, mask, mask_labels)
                if hne_features is None:
                    continue
                record.update(hne_features)

            if mask is not None:
                composition = tile_class_composition(mask, tformed_quad, mask_labels)
                for class_name in mask_labels.values():
                    record[f'pct_{class_name}'] = (
                        composition.get(class_name, 0.0) if composition is not None else None
                    )

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

            if show_tiles:
                tile_render_data.append({
                    'czi_patch': czi_patch,
                })

                fig, axes = plt.subplots(1, 1)
                axes.imshow(np.rot90(czi_patch, k=3))
                axes.set_title('Original H&E Patch')
                plt.show()


        # To save tile origins to pull into qupath->
        # originarray= np.array(transformed_originList)
        # tiles_to_geojson(originarray, output_geojson_path)

        if records:
            tile_labels_df = pd.DataFrame(records)
            tile_labels_df.to_csv(output_tile_labels_csv_path, index=False)
            print(f'Saved {len(tile_labels_df)} tile locations with annotation class composition to {output_tile_labels_csv_path}')
