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
    print(file_numb)
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

if __name__ == '__main__':
    reg_InV_path = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/InVivo_MR/RegDataOut/InVivoRegToBlockFace_0310-1340.nii.gz'
    reg_HnE_path = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/HnE/Block06/HnEBlock06Reg.nii'

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

    # Resample MRI into H&E voxel grid using affine information.
    # inv(MRI_affine) @ HnE_affine maps H&E voxel coords → MRI voxel coords,
    # which is the coordinate transform scipy.ndimage.affine_transform expects.
    hne_to_mri = np.linalg.inv(reg_InV_affine) @ reg_HnE_affine
    matrix = hne_to_mri[:3, :3]
    offset = hne_to_mri[:3, 3]
    output_shape = reg_HnE_arr.shape[:3]

    reg_InV_resampled = affine_transform(
        reg_InV_arr, matrix, offset=offset, output_shape=output_shape, order=1
    )
    print('Resampled MRI shape:', reg_InV_resampled.shape)

    tilesize=50
    for i in range(1):
        i=3
        hne_ds_im= reg_HnE_arr[:,:,i,:]
        origin_list = tiling_tool(reg_HnE_arr[:,:,i,:], tilesize)

        file_path, landmarks = load_landmarks(reg_HnE_path, i)
        scale_fac=20
        src = np.array([[p.y(), p.x()] for p in landmarks[0]])*3  # fixed
        dst = np.array([[p.x(), p.y()] for p in landmarks[1]])*3  # H&E
        splines = ski.transform.ThinPlateSplineTransform.from_estimate(src, dst)
        splines_inv = ski.transform.ThinPlateSplineTransform.from_estimate(dst, src)

        CZI_filepath= get_CZI(file_path)
        czifile = CziFile(CZI_filepath)
        bbox = czifile.get_mosaic_bounding_box()

        #output_image = ski.transform.warp(hne_ds_im, splines_inv, output_shape=(czi_img.shape[0]*scale_fac, czi_img.shape[1]*scale_fac, czi_img.shape[2]))
        # normalize and convert to uint8
        #output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)

        #Tile-Wise work through-
        for i in range(len(origin_list)):
            O_up= origin_list[i]
            quadlist= np.array([O_up, [O_up[0]+tilesize, O_up[1]], [O_up[0], O_up[1]+tilesize], [O_up[0]+tilesize ,O_up[1]+tilesize]])
            print(quadlist)
            tformed_quad= splines(quadlist[:, ::-1])[:, ::-1]*scale_fac
            print(tformed_quad)
            # Bounding box in CZI space
            r_min, c_min = tformed_quad.min(axis=0).astype(int)
            r_max, c_max = tformed_quad.max(axis=0).astype(int)
            h = r_max - r_min
            w = c_max - c_min
            # Load just that region from CZI
            region = (bbox.x + c_min, bbox.y + r_min, w, h)
            print("r_min:", r_min)
            print("c_min:", c_min)
            print("w:", w)
            print("h:", h)
            print("region:", region)

            czi_patch = czifile.read_mosaic(C=0, scale_factor=1, region=region)[0]
            print("czi_patch:", czi_patch.shape)

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(hne_ds_im)
            axes[0].scatter(quadlist[:,1], quadlist[:,0], c='r', marker='o', s=5)
            axes[1].imshow(czi_patch)
            axes[1].scatter(tformed_quad[:,1]-c_min, tformed_quad[:,0]-r_min, c='r', marker='o', s=5)
            plt.show()



    # app = QApplication(sys.argv)
    # viewer_reg = VolumeViewer(norm255(reg_InV_resampled),  reg_HnE_arr,
    #                           'Registered InVivo — HnE')
    # viewer_reg.show()
    # sys.exit(app.exec_())
