from PyQt5.QtWidgets import QApplication
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

def load_landmarks(hne_path, index):
    """Load landmarks for slice at `index` from the Landmarks folder next to hne_path."""
    landmarks_dir = os.path.join(os.path.dirname(hne_path), 'Landmarks')
    files = sorted(f for f in os.listdir(landmarks_dir)
                   if f.endswith('_landmarks.npy') and not f.startswith('._'))
    filepath=os.path.join(landmarks_dir, files[index])
    return (filepath, np.load(filepath, allow_pickle=True))

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

def get_CZI(filepath):
    parts = filepath.split(os.sep)
    rabbit_id = parts[10]
    block = parts[12]
    file_numb=re.search(r'\d+', parts[-1]).group()
    print("file numb", file_numb)
    CZI_filepath = CSZ_CZI_lookup(rabbit_id, block, file_numb)
    return CZI_filepath

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
    reg_InV_path = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToBlock07_0402-0958.nii.gz'
    reg_HnE_path = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/HnE/Block07/HnEBlock07.nii'

    # --- Annotation config ---
    # Set mask_path to the TIFF output of WorkingGeoJson.py, or None if no annotations available.
    # mask_labels values must match the tisslabels dict in WorkingGeoJson.py; 0 = unannotated (Muscle).
    mask_path = '/Users/jbonaventura/Desktop/Annotations/HnE_R23-055_H7_7a_annotations_Mask.tiff' # Can also be set to None if None!
    mask_labels = {0: 'Muscle', 100: 'Necrotic Tissue', 200: 'Immune Infiltration'}  # add entries e.g. {0: 'Muscle', 10: 'Necrotic Tissue', 20: 'Immune Infiltration'}
    show_tiles = False  # set True to visualize each tile during extraction

    annotations_dir = os.path.dirname(mask_path) if mask_path is not None else '/Users/jbonaventura/Desktop/Annotations'
    slide_id = re.search(r'HnE_(.+?)_annotations', os.path.basename(mask_path)).group(1) if mask_path is not None else 'unknown'
    output_csv_path = os.path.join(annotations_dir, f'{slide_id}_features.csv')
    output_geojson_path = os.path.join(annotations_dir, f'{slide_id}_tiles.geojson')

    reg_InV = nib.load(reg_InV_path)
    reg_InV_affine = reg_InV.affine
    reg_InV_arr = reg_InV.get_fdata()

    reg_HnE = nib.load(reg_HnE_path)
    reg_HnE_affine = reg_HnE.affine
    raw = np.asarray(reg_HnE.dataobj)
    # RGB NIfTI is stored as a structured array with named fields (R, G, B)
    reg_HnE_arr = np.stack([raw[name] for name in raw.dtype.names], axis=-1)

    print(reg_HnE_arr.shape)
    print(reg_InV_arr.shape)
    print(reg_InV_affine)
    print(reg_HnE_affine)

    mask = np.array(Image.open(mask_path)) if mask_path is not None else None

    # Resample MRI into H&E voxel grid using affine information.
    # inv(MRI_affine) @ HnE_affine maps H&E voxel coords → MRI voxel coords,
    # which is the coordinate transform scipy.ndimage.affine_transform expects.
    hne_to_mri = np.linalg.inv(reg_InV_affine) @ reg_HnE_affine
    matrix = hne_to_mri[:3, :3]
    offset = hne_to_mri[:3, 3]
    output_shape = reg_HnE_arr.shape[:3]

    reg_InV_resampled = affine_transform(
        reg_InV_arr, matrix, offset=offset, output_shape=output_shape, order=0
    )
    print('Resampled MRI shape:', reg_InV_resampled.shape)

    tilesize=50
    for i in range(1):
        i=5
        hne_ds_im= reg_HnE_arr[:,:,i,:]
        MR_Slice = reg_InV_resampled[:,:,i]
        print("MR_Slice info", np.max(MR_Slice), np.min(MR_Slice))

        origin_list = tiling_tool(reg_HnE_arr[:,:,i,:], tilesize)

        file_path, landmarks = load_landmarks(reg_HnE_path, i)
        print("file_path", file_path)
        scale_fac=20
        src = np.array([[p.y(), p.x()] for p in landmarks[0]])*3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]])*3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)
        splines_inv = ski.transform.ThinPlateSplineTransform.from_estimate(dst, src)

        CZI_filepath= get_CZI(file_path)
        print(CZI_filepath)
        czifile = CziFile(CZI_filepath)
        bbox = czifile.get_mosaic_bounding_box()
        # czi_patch = czifile.read_mosaic(C=0, scale_factor=1/10, region=(bbox.x, bbox.y, bbox.w, bbox.h))[0]
        # czi_patch[:, :, [0, 2]] = czi_patch[:, :, [2, 0]]

        #output_image = ski.transform.warp(hne_ds_im, splines_inv, output_shape=(czi_img.shape[0]*scale_fac, czi_img.shape[1]*scale_fac, czi_img.shape[2]))
        # normalize and convert to uint8
        #output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)
        # tformed_origins = splines(origin_list[:, ::-1])[:, ::-1]* 2


        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(hne_ds_im)
        #axes[0].scatter(origin_list[:,1], origin_list[:,0], c='r', marker='o', s=5)
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle(
                (col, row),  # note: matplotlib uses (x, y) = (col, row)
                50, 50,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)

        axes[1].imshow(MR_Slice, cmap='gray')
        #axes[1].scatter(origin_list[:,1], origin_list[:,0], c='r', marker='o', s=5)
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle(
                (col, row),  # note: matplotlib uses (x, y) = (col, row)
                50, 50,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            axes[1].add_patch(rect)
        plt.show()

        #Tile-Wise work through-
        records = []
        print(origin_list[:5])

        MRFeatures= np.zeros((len(origin_list), 19))
        MRFeatures[:,:2]=origin_list

        transformed_originList=[]
        for i in range(len(origin_list)):
        #for i in range(3):
            O_up= origin_list[i]
            quadlist= np.array([O_up, [O_up[0]+tilesize, O_up[1]], [O_up[0]+tilesize, O_up[1]+tilesize], [O_up[0] ,O_up[1]+tilesize]])
            tformed_quad= splines(quadlist[:, ::-1])[:, ::-1]*scale_fac
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
            record = {'tile_id': i, 'origin_row': O_up[0], 'origin_col': O_up[1],
                      'czi_r_min': czi_r_min, 'czi_c_min': czi_c_min,
                      'czi_r_max': czi_r_max, 'czi_c_max': czi_c_max}

            patch_f = czi_patch.astype(float)
            patch_f[patch_f == 0] = np.nan
            flat = patch_f.reshape(-1, 3)
            means = np.nanmean(flat, axis=0)
            stds = np.nanstd(flat, axis=0)
            record.update({'mean_R': means[0], 'mean_G': means[1], 'mean_B': means[2],
                           'std_R': stds[0], 'std_G': stds[1], 'std_B': stds[2]})

            haralick_names = ['asm', 'contrast', 'correlation', 'variance', 'homogeneity',
                              'sum_avg', 'sum_var', 'sum_entropy', 'entropy',
                              'diff_var', 'diff_entropy', 'energy', 'dissimilarity', 'max_prob']
            redchan = czi_patch[:, :, 0].astype(np.uint8)
            record.update({f'red_{n}': v for n, v in zip(haralick_names, haralick_features(redchan))})
            bluechan = czi_patch[:, :, 2].astype(np.uint8)
            record.update({f'blue_{n}': v for n, v in zip(haralick_names, haralick_features(bluechan))})

            if mask is not None:
                label = label_tile_from_mask(mask, tformed_quad, mask_labels)
                if label is None:
                    continue  # skip ambiguous tiles
                record['label'] = label

            records.append(record)

            # #Masking to only get polygon->
            # verticies = tformed_quad[:,::-1].copy()
            # verticies -= [c_min, r_min]
            # verticies = verticies.astype(np.int32)
            #
            # #Mask out bright background->
            # tilewarp = np.zeros(czi_patch.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(tilewarp, [verticies], 255)
            # czi_patch = czi_patch * (tilewarp[:, :, np.newaxis] > 0)
            # #Mask out bright background->
            # rgbmean = np.mean(czi_patch, axis=2)
            # whiteIm = np.where(rgbmean > 210, 0, 1)
            # czi_patch = czi_patch * whiteIm[:, :, np.newaxis]


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
            # HnEFeatures[i, 2:5]= np.nanmean(np.nanmean(nancopy, axis=0), axis=0)

            #Texture features-
            #1st convert patch to greyscale->
            # Apply to red and green color channels seperately->
            # redchan= czi_patch[:,:,0].astype(np.uint8)
            # textfeatures_red = haralick_features(redchan)
            # HnEFeatures[i, 5:19] = textfeatures_red
            #
            # bluechan= czi_patch[:,:,2].astype(np.uint8)
            # textfeatures_blue = haralick_features(bluechan)
            # HnEFeatures[i, 19:33] = textfeatures_blue


            # for s in range(5):
            #     start= 33+ s*28
            #     redchan = ds_list[s][:, :, 0].astype(np.uint8)
            #     textfeatures_red = haralick_features(redchan)
            #     HnEFeatures[i, start:start+14] = textfeatures_red
            #
            #     bluechan = ds_list[s][:, :, 2].astype(np.uint8)
            #     textfeatures_blue = haralick_features(bluechan)
            #     HnEFeatures[i, start+14:start+28] = textfeatures_blue
            #
            # #Okay, now we want to extend this to the MR data!
            # #pull cooresponding tile from MRSlice->
            MR_Tile= MR_Slice[O_up[0]:O_up[0]+tilesize, O_up[1]:O_up[1]+tilesize]
            # dy, dx = np.gradient(MR_Tile)
            #
            # MRFeatures[i, 2] = np.mean(MR_Tile)
            # MRFeatures[i, 3] = np.std(MR_Tile)
            # MRFeatures[i, 4] = np.mean(dx)
            # MRFeatures[i, 5] = np.mean(dy)

            # print(MRFeatures[i, :6])

            if show_tiles:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(np.fliplr(np.rot90(czi_patch, k=2)))
                axes[0].set_title('Original H&E')
                axes[1].imshow(MR_Tile, cmap='gray', interpolation='none', vmin=0, vmax=2176)
                axes[1].set_title("MR Tile")
                plt.tight_layout()
                plt.show()

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

        # print(transformed_originList)
        originarray= np.array(transformed_originList)
        tiles_to_geojson(originarray, output_geojson_path)

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
    # viewer_reg = VolumeViewer(norm255(reg_InV_resampled),  reg_HnE_arr,
    #                           'Registered InVivo — HnE')
    # viewer_reg.show()
    # sys.exit(app.exec_())
