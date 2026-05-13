from PyQt5.QtWidgets import QApplication

#from CZItoTiff import block_no
from Viewer import VolumeViewer
import sys
import os
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import affine_transform
import pandas as pd
from aicspylibczi import CziFile
import skimage as ski
from skimage.measure import find_contours
from skimage.morphology import dilation, square
import json
import cv2
import glob
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import block_reduce
import torch
from PIL import Image
from TileUtils import tiling_tool, load_landmarks, get_bf_slice_index, CSZ_CZI_lookup
from shapely.geometry import Polygon
from HnEFeatureExtraction import extract_features


def rectilinearize(pts, edge_map):
    result = []
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        result.append(p1)
        dx, dy = int(p2[0] - p1[0]), int(p2[1] - p1[1])
        if dx != 0 and dy != 0:
            mid_h = [p1[0] + dx, p1[
                1]]  # horizontal first
            mid_v = [p1[0], p1[1] + dy]  # vertical first
            # go through whichever intermediate point is in the border
            in_border = lambda p: edge_map[int(p[1]), int(p[0])] > 0
            result.append(mid_h if not in_border(mid_h) else mid_v)
    return np.array(result)

if __name__ == '__main__':
    Rabbit = 'R23-055'
    block_no = 7
    Block = 'Block0' + str(block_no)
    #Need this one to draw tiles-
    #Need to call directory instead of niftis->
    root_dir = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
    rabbase= os.path.join(root_dir, Rabbit)
    hne_base_dir = os.path.join(rabbase, 'HnE', Block)

    #Change here to change (Should work with any of the niftis registered to BlockFace)-
    reg_HnE_dir = os.path.join(hne_base_dir, 'Registered')
    #Path to voxel map-
    reg_show = os.path.join(rabbase, 'InVivo_MR/RegDataOut/Voxel_Maps/voxel_tile_vol_reg_to_'+Block+'.nii.gz')

    bf_cropped_dir = os.path.join(rabbase, 'BlockFace_RGB', Block,'CroppedImages')
    print(bf_cropped_dir)

    reg_InV = nib.load(reg_show)
    reg_InV_affine = reg_InV.affine
    reg_InV_arr = reg_InV.get_fdata()
    if reg_InV.affine[0, 0] > 0 and reg_InV.affine[1, 1] > 0:
        reg_InV_arr = reg_InV_arr[::-1, ::-1, :]

    mr_volumes = {}

    mr_volumes_dir = os.path.join(rabbase, 'InVivo_MR/RegDataOut/Day3End_Registered_RegTo'+Block)
    for _f in sorted(glob.glob(os.path.join(mr_volumes_dir, '*.nii.gz'))):
        _col_name = '_'.join(os.path.basename(_f).replace('.nii.gz', '').split('_')[:2])
        _nib = nib.load(_f)
        _arr = _nib.get_fdata(dtype=np.float32)
        if _nib.affine[0, 0] > 0 and _nib.affine[1, 1] > 0:
            _arr = _arr[::-1, ::-1, :]
        mr_volumes[_col_name] = _arr

    mr_intensity_arr = next(v for k, v in mr_volumes.items() if k.startswith('InVivoReg'))

    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir) if f.endswith('Reg.png') and not f.startswith('._'))
    hne_bf_indices = [re.search(r'\d+', f).group()[-2:] for f in hne_filenames]
    hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in hne_filenames]
    reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)

    print(reg_HnE_arr.shape)
    print(reg_InV_arr.shape)
    print(hne_bf_indices[0])

    Save_Voxel_Geoms = True

    tilesize=50
    chunk_size_ds = 200   # non-overlapping chunk size in downsampled HnE pixels
    chunk_buffer_full = 1000  # buffer in full-res pixels added around each chunk read
    poly_id_counter = 0
    features_dir = os.path.join(rabbase, 'Analysis', Block)
    os.makedirs(features_dir, exist_ok=True)

    for img in range(6):
        all_results = []
        #i=5
        hne_ds_im= reg_HnE_arr[:,:,img,:]
        hne_Name = hne_bf_indices[img]

        #Finding tps transform between CZI and blockface->
        img_number = re.search(r'\d+', hne_filenames[img]).group()  # e.g. '0011'
        landmarks = load_landmarks(hne_base_dir, img_number)
        CZI_filepath = CSZ_CZI_lookup(Rabbit, Block, img_number)

        scale_fac=20 #scale factor between HnE raw and downsampled
        src = np.array([[p.y(), p.x()] for p in landmarks[0]])*3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]])*3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)
        splines_inv = ski.transform.ThinPlateSplineTransform.from_estimate(dst, src)

        #Use blockface filename position in CroppedImages to find the correct NIfTI slice-
        slice_num = get_bf_slice_index(bf_cropped_dir, img_number)
        print(slice_num)
        #Insead of resampling we can use offset to select the correct slice-
        MR_Slice = reg_InV_arr [:,:,slice_num].T
        #MR_Slice = reg_InV_resampled[:,:,i]
        print("MR_Slice info", np.max(MR_Slice), np.min(MR_Slice))
        #Upsampling by 4 this is for visualization and tile extraction purposes- not for analysis-
        vol_upsampled = np.repeat(np.repeat(MR_Slice, 4, axis=0), 4, axis=1)
        MR_Slice_us= vol_upsampled[:hne_ds_im.shape[0],:hne_ds_im.shape[1]]

        print(hne_ds_im.shape, MR_Slice_us.shape)

        origin_list = tiling_tool(reg_HnE_arr[:,:,img,:], tilesize)
        tile_mask = np.zeros(MR_Slice_us.shape[:2], dtype=np.uint8)
        for origin in origin_list:
            row, col = origin
            tile_mask[row:row + tilesize, col:col + tilesize] = 1

        n_cols = 1 + len(mr_volumes)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 8))
        axes[0].imshow(hne_ds_im)
        axes[0].contour(tile_mask, levels=[0.5], colors='white', linewidths=0.5)
        axes[0].set_title('H&E', fontsize=8)
        axes[0].axis('off')
        for ax, (vol_name, vol_arr) in zip(axes[1:], mr_volumes.items()):
            _slice = vol_arr[:, :, slice_num].T
            _us = np.repeat(np.repeat(_slice, 4, axis=0), 4, axis=1)
            _us = _us[:hne_ds_im.shape[0], :hne_ds_im.shape[1]]
            ax.imshow(_us, cmap='gray')
            ax.contour(tile_mask, levels=[0.5], colors='white', linewidths=0.5)
            ax.set_title(vol_name, fontsize=8)
            ax.axis('off')
        plt.suptitle(f'Slice {hne_Name}')
        plt.tight_layout(pad=1)
        plt.show()


        #Convert  MR to Gradient space-
        dy, dx = np.gradient(MR_Slice_us.astype(float))
        edge_map = (np.abs(np.sign(dy)) + np.abs(np.sign(dx))).astype(float)
        interior = (~(edge_map > 0)).astype(np.uint8)
        interior_masked = interior & tile_mask

        contours, _ = cv2.findContours(interior_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expanded_contours = []
        hne_polys = []
        poly_ids = []
        bf_centroids = []
        n_invalid = 0
        for contour in contours:
            pts = contour[:, 0, :]  # (N, 2) x,y
            pts = rectilinearize(pts, edge_map)
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            bf_centroid_row = int(poly.centroid.y)
            bf_centroid_col = int(poly.centroid.x)
            expanded = poly.buffer(1.5, join_style=2)
            expanded_contours.append(expanded)
            bf_pts = np.array(expanded.exterior.coords)  # (N, 2) [x, y] blockface
            hne_pts = splines(bf_pts[:, ::-1])  # [x,y]→[row,col]→splines→[x,y] downsampled HnE
            hne_poly = Polygon(hne_pts)
            if not hne_poly.is_valid:
                hne_poly = hne_poly.buffer(0)
                if hne_poly.geom_type == 'MultiPolygon':
                    hne_poly = max(hne_poly.geoms, key=lambda g: g.area)
                if not hne_poly.is_valid:
                    n_invalid += 1
                    continue
            hne_polys.append(hne_poly)
            poly_ids.append(poly_id_counter)
            bf_centroids.append((bf_centroid_row, bf_centroid_col))
            poly_id_counter += 1
        print(f"Slice {img}: {len(hne_polys)} valid polygons, {n_invalid} dropped as invalid")

        centroids_ds = np.array([list(p.centroid.coords[0]) for p in hne_polys])

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        mr_intensity_slice = mr_intensity_arr[:, :, slice_num].T
        mr_intensity_us = np.repeat(np.repeat(mr_intensity_slice, 4, axis=0), 4, axis=1)
        mr_intensity_us = mr_intensity_us[:hne_ds_im.shape[0], :hne_ds_im.shape[1]]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(mr_intensity_us, cmap='gray')
        for i, poly in enumerate(expanded_contours):
            x, y = poly.exterior.xy
            ax.plot(x, y, '-', color=colors[i % 10], linewidth=0.5)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


        czifile = CziFile(CZI_filepath)
        bbox = czifile.get_mosaic_bounding_box()

        print("BBOXES- ",bbox.x, bbox.y)

        # If we want to look at a downsampled CZI file- useful for verifying splines transform is acting how we want-
        hne_full_ds = czifile.read_mosaic(C=0, scale_factor=1 / 20, region=(bbox.x, bbox.y, bbox.w, bbox.h))[0]
        hne_full_ds[:, :, [0, 2]] = hne_full_ds[:, :, [2, 0]]  # BGR→RGB

        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(hne_full_ds)
        # for i, poly in enumerate(hne_polys):
        #     x, y = poly.exterior.xy
        #     ax.plot(x, y, '-', color=colors[i % 10], linewidth=0.5)
        # ax.axis('off')
        # plt.tight_layout()
        # plt.show()


        # --- Feature extraction ---
        max_radius_ds = 0
        for poly in hne_polys:
            cx, cy = poly.centroid.x, poly.centroid.y
            px, py = poly.exterior.xy
            radii = np.sqrt((np.array(px) - cx) ** 2 + (np.array(py) - cy) ** 2)
            max_radius_ds = max(max_radius_ds, radii.max())
        chunk_buffer_full = int(max_radius_ds * scale_fac) + 50

        x_edges = np.arange(centroids_ds[:, 0].min(), centroids_ds[:, 0].max() + chunk_size_ds, chunk_size_ds)
        y_edges = np.arange(centroids_ds[:, 1].min(), centroids_ds[:, 1].max() + chunk_size_ds, chunk_size_ds)

        for xi in range(len(x_edges) - 1):
            for yi in range(len(y_edges) - 1):
                cx0, cx1 = x_edges[xi], x_edges[xi + 1]
                cy0, cy1 = y_edges[yi], y_edges[yi + 1]

                in_chunk = []
                for p, pid, bfc, c in zip(hne_polys, poly_ids, bf_centroids, centroids_ds):
                    if cx0 <= c[0] < cx1 and cy0 <= c[1] < cy1:
                        in_chunk.append((p, pid, bfc))
                if not in_chunk:
                    continue

                print(len(in_chunk), "Voxels in chunk")

                rx  = max(bbox.x, int(cx0 * scale_fac) + bbox.x - chunk_buffer_full)
                ry  = max(bbox.y, int(cy0 * scale_fac) + bbox.y - chunk_buffer_full)
                rx_end = min(bbox.x + bbox.w, int(cx1 * scale_fac) + bbox.x + chunk_buffer_full)
                ry_end = min(bbox.y + bbox.h, int(cy1 * scale_fac) + bbox.y + chunk_buffer_full)
                chunk_img = czifile.read_mosaic(C=0, scale_factor=1, region=(rx, ry, rx_end - rx, ry_end - ry))[0]
                chunk_img[:, :, [0, 2]] = chunk_img[:, :, [2, 0]]  # BGR→RGB

                # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                # ax.imshow(chunk_img)
                # for i, (poly, _, _) in enumerate(in_chunk):
                #     px, py = poly.exterior.xy
                #     ax.plot(np.array(px) * scale_fac + bbox.x - rx,
                #             np.array(py) * scale_fac + bbox.y - ry,
                #             '-', color=colors[i % 10], linewidth=1)
                # ax.axis('off')
                # plt.tight_layout()
                # plt.show()

                for poly, pid, (centroid_row, centroid_col) in in_chunk:
                    px, py = poly.exterior.xy
                    px_patch = np.array(px) * scale_fac + bbox.x - rx
                    py_patch = np.array(py) * scale_fac + bbox.y - ry

                    x0 = max(0, int(min(px_patch)) - 1)
                    y0 = max(0, int(min(py_patch)) - 1)
                    x1 = min(chunk_img.shape[1], int(max(px_patch)) + 2)
                    y1 = min(chunk_img.shape[0], int(max(py_patch)) + 2)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    verts = np.stack([px_patch, py_patch], axis=1).astype(np.int32)
                    mask = np.zeros(chunk_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [verts], 255)

                    patch_crop = chunk_img[y0:y1, x0:x1]
                    mask_crop = mask[y0:y1, x0:x1]
                    if mask_crop.sum() == 0:
                        continue

                    masked_pixels = patch_crop[mask_crop > 0]
                    bg_fraction = (masked_pixels.mean(axis=1) > 210).mean()
                    if bg_fraction > 0.3:
                        continue

                    masked_patch = patch_crop.copy()
                    masked_patch[mask_crop == 0] = 0
                    # plt.imshow(masked_patch)
                    # plt.show()

                    _ref_arr = next(iter(mr_volumes.values()))
                    mr_row = np.clip(centroid_row // 4, 0, _ref_arr.shape[1] - 1)
                    mr_col = np.clip(centroid_col // 4, 0, _ref_arr.shape[0] - 1)

                    result = {'poly_id': pid, 'centroid_row': centroid_row, 'centroid_col': centroid_col}
                    for vol_name, vol_arr in mr_volumes.items():
                        result[vol_name] = vol_arr[mr_col, mr_row, slice_num]
                    result.update(extract_features(masked_patch))
                    all_results.append(result)


        df = pd.DataFrame(all_results)
        czi_stem = os.path.splitext(os.path.basename(CZI_filepath))[0]
        csv_path = os.path.join(features_dir, f'voxel_features_{czi_stem}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")

        if Save_Voxel_Geoms:
            surviving_ids = set(df['poly_id'])
            geojson_features = []
            for pid, poly in zip(poly_ids, hne_polys):
                if pid not in surviving_ids:
                    continue
                coords = np.array(poly.exterior.coords)
                coords[:, 0] = coords[:, 0] * scale_fac
                coords[:, 1] = coords[:, 1] * scale_fac
                ring = [[float(x), float(y)] for x, y in coords]
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"object_type": "annotation", "poly_id": int(pid)}
                })
            geojson_path = os.path.join(features_dir, f'voxel_polygons_{czi_stem}.geojson')
            with open(geojson_path, 'w') as f:
                json.dump({"type": "FeatureCollection", "features": geojson_features}, f)
            print(f"Saved {len(geojson_features)} polygons to {geojson_path}")
