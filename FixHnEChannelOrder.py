import os
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

# Rewrites one block's 1/20 H&E tifs and Registered pngs in place with R and B swapped.
# Files made before the CZI BGR->RGB fix (swap_channel_order in CZItoTiff.py) have red and
# blue reversed. The TPS warp in gui_he2blk.py treats each channel independently, so swapping
# channels on an existing Reg.png gives exactly the image a fresh CZItoTiff + TPS would make.
# NOTE: no re-run guard - running a block twice swaps the colours back.

rab_ID = 'R24-103'
block_no = 'Block04'

BaseSavePath = "//System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData"
block_num_str = re.findall(r'\d+', block_no)[0]
block_dirpath = os.path.join(BaseSavePath, rab_ID, "HnE", f"Block{block_num_str}")
registered_dirpath = os.path.join(block_dirpath, "Registered")


def swap_red_blue(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        print(f"SKIPPING {path}: mode {img.mode}, expected RGB")
        return
    Image.fromarray(np.array(img)[..., ::-1]).save(path)
    print(f"Swapped {path}")


tif_files = [f for f in os.listdir(block_dirpath)
             if f.lower().startswith('hne_img') and f.lower().endswith('.tif')]
reg_files = [f for f in os.listdir(registered_dirpath)
             if f.lower().endswith('reg.png') and not f.startswith('._')]

paths = ([os.path.join(block_dirpath, f) for f in sorted(tif_files)] +
         [os.path.join(registered_dirpath, f) for f in sorted(reg_files)])
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(swap_red_blue, paths))  # list() makes any worker exception raise here

print(f"Done: {len(tif_files)} tifs, {len(reg_files)} registered pngs in {block_dirpath}")
