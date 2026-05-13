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
    return tilesarray

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
    # Path to CSV-
    csv_dirpath = os.path.join(BaseCephPath, rab_ID, rab_ID + "_BlockFaceImages", block_no, "csv_files")
    #Find last CSV from that directory->
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


if __name__ == '__main__':
    #Need this one to draw tiles-
    #Need to call directory instead of niftis->

    hne_base_dir = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/HnE/Block07'
    #Change here to change (Should work with any of the niftis registered to BlockFace-
    reg_HnE_dir = os.path.join(hne_base_dir, 'Registered')
    reg_show = '/Users/jbonaventura/Desktop/Annotations/voxel_tile_vol_reg_to_Block07.nii.gz'
    #Could be nice to load them all here so they can be toggled through in the viewer- I.E. click side to side and then see Im?
    rabbase = os.path.split(os.path.split(hne_base_dir)[0])[0]
    bf_cropped_dir = os.path.join(rabbase, 'BlockFace_RGB', 'Block07','CroppedImages')

    reg_InV = nib.load(reg_show)
    reg_InV_affine = reg_InV.affine
    reg_InV_arr = reg_InV.get_fdata()

    hne_filenames = sorted(f for f in os.listdir(reg_HnE_dir) if f.endswith('Reg.png') and not f.startswith('._'))
    hne_bf_indices = [re.search(r'\d+', f).group()[-2:] for f in hne_filenames]
    hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in hne_filenames]
    reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)

    print(reg_HnE_arr.shape)
    print(reg_InV_arr.shape)
    print(hne_bf_indices[0])


    hne_parts = hne_base_dir.split(os.sep)
    rabbit_id = hne_parts[10]
    block = hne_parts[12]

    tilesize=100
    for i in range(6):
        #i=5
        hne_ds_im= reg_HnE_arr[:,:,i,:]
        hne_Name = hne_bf_indices[i]

        img_number = re.search(r'\d+', hne_filenames[i]).group()  # e.g. '0011'
        landmarks = load_landmarks(hne_base_dir, img_number)
        CZI_filepath = CSZ_CZI_lookup(rabbit_id, block, img_number)

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

        origin_list = tiling_tool(reg_HnE_arr[:,:,i,:], tilesize)

        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        axes[0].imshow(hne_ds_im)
        #axes[0].scatter(origin_list[:,1], origin_list[:,0], c='r', marker='o', s=5)
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle((col, row), tilesize, tilesize, linewidth=1, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)

        axes[1].imshow(MR_Slice_us, cmap='gray')
        #axes[1].scatter(origin_list[:,1], origin_list[:,0], c='r', marker='o', s=5)
        for q in range(len(origin_list)):
            row = origin_list[q,0]
            col = origin_list[q,1]
            color = 'black'
            rect = patches.Rectangle((col, row), tilesize, tilesize, linewidth=1, edgecolor=color, facecolor='none')
            axes[1].add_patch(rect)
        for ax in axes:
            ax.axis('off')
        plt.tight_layout(pad=1)
        plt.show()

