from aicspylibczi import CziFile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import pandas as pd
import math
import ast

#Now to add pointing so we can go from rabbit number and block to all of this-
rab_ID='R23-292'
block_no='block07'

# #Path to Savepoint-
BaseSavePath="/Users/jbonaventura/Downloads/"
save_dirpath=os.path.join(BaseSavePath, rab_ID, "HnE", block_no)
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)

#With this we can do-
BaseCephPath="/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/"
#Path to CSV-
csv_dirpath=os.path.join(BaseCephPath, rab_ID, rab_ID + "_BlockFaceImages",block_no,"csv_files")
files = [f for f in os.listdir(csv_dirpath) if f.endswith('csv')]
sorted_filenames = sorted(files, key=lambda x: float(re.findall(r'\d+', x)[0]))
csv_file=sorted_filenames[-1]
csv_filepath=os.path.join(csv_dirpath, csv_file)
#Load in the csv file to get the proper term to rename czi files-
df = pd.read_csv(csv_filepath)
csv_array = df.to_numpy()
match_vals=[]
for i in range(csv_array.shape[0]):
    if not pd.isna(csv_array[i,2]):
        for val in csv_array[i, 2].split(';'):
            match_vals.append([val,ast.literal_eval(csv_array[i,0])[0].split('_')[1]])
        #Sort out which images have HnE slides and grab image number with HnE index-
        #match_vals.append([csv_array[i,2].split(';'), ast.literal_eval(csv_array[i,0])[0].split('_')[1]])
match_array = np.array(match_vals)

#Path to CZI directory-
czi_dirpath=os.path.join(BaseCephPath, rab_ID, rab_ID + "_HnE_5x", block_no)
for file in os.listdir(czi_dirpath):
    #Can edit for funky naming stuff that may go on-
    if "HnE_R23-292" in file:
        print(file)
        full_path = os.path.join(czi_dirpath, file)
        file_name = os.path.splitext(os.path.basename(full_path))[0]
        HnE_Label = file_name.split("_")[-2]
        print(HnE_Label)
        idx = np.where(match_array==HnE_Label)[0]
        try:
            image_tag = str(match_array[idx,1][0])
        except:
            HnE_Label = file_name.split("_")[-1]
            print(HnE_Label)
            idx = np.where(match_array == HnE_Label)[0]
            image_tag = str(match_array[idx, 1][0])
        print(image_tag)

        czifile = CziFile(full_path)

        bbox = czifile.get_mosaic_bounding_box()
        czi_img = czifile.read_mosaic(C=0, scale_factor=1/20, region=(bbox.x, bbox.y, bbox.w, bbox.h), background_color=(1,1,1))[0,:,:,:]
        plt.imshow(czi_img)
        plt.show()

        #Save Image as Tiff-
        new_file_path = os.path.join(save_dirpath, "HnE_IMG_"+ image_tag + ".tif")
        CImage = Image.fromarray(czi_img)
        CImage.save(new_file_path, 'TIFF')


