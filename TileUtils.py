import os
import re
import numpy as np
import pandas as pd


def tiling_tool(twoDIm, tile_size):
    #Mask out background->
    rgbmean = np.mean(twoDIm, axis=2)
    whiteIm=np.where(rgbmean>210,0,1)
    twoDIm= twoDIm * whiteIm[:, :, np.newaxis]
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
    czi_file = next(f for f in os.listdir(czi_dirpath) if re.search(r'(?<!\d)' + re.escape(czi_name), f))
    czi_path = os.path.join(czi_dirpath, czi_file)
    return czi_path
