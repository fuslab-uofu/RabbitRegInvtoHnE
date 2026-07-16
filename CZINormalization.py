import os
import numpy as np
from aicspylibczi import CziFile
import tifffile
import torchstain

from TileUtils import make_tissue_mask

#--- config: pick whichever rabbit/block you want to test on ---
rab_ID = "R24-101"
block_no = "block12"
n_test_files = 3          # just grab a handful for the sanity check
scale_factor = 1 / 20     # matches CZItoTiff.py's existing downsample convention

REFERENCE_CZI = "/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/R23-055/R23-055_HnE_5x/block07/HnE_R23-055_H7_5a_5X.czi"
BaseCephPath = "/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/"
TEST_OUTPUT_DIR = "/Users/jbonaventura/Documents/GitHub/RabbitRegInvtoHnE/NormalizationSanityCheck"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

czi_dirpath = os.path.join(BaseCephPath, rab_ID, f"{rab_ID}_HnE_5X", block_no)
czi_files = [f for f in os.listdir(czi_dirpath) if f.lower().endswith(".czi")][:n_test_files]


def read_czi_rgb(czifile, scale_factor=1 / 20):
    bbox = czifile.get_mosaic_bounding_box()
    img = czifile.read_mosaic(
        C=0, scale_factor=scale_factor,
        region=(bbox.x, bbox.y, bbox.w, bbox.h),
        background_color=(1, 1, 1),
    )[0, :, :, :]
    return img


def get_pixel_size_um(czifile):
    # CZI stores raw (full-res) pixel size in meters under Scaling/Items/Distance
    value = czifile.meta.find('.//Scaling/Items/Distance[@Id="X"]/Value')
    return float(value.text) * 1e6


def blank_background(img):
    # Force background/debris to pure white *before* normalization, so it's excluded from
    # stain-vector estimation and the concentration percentile calc, not just reconstructed
    # as noise afterward.
    tissue_mask = make_tissue_mask(img)
    img = img.copy()
    img[tissue_mask == 0] = 255
    return img


#--- fit normalizer directly off the reference CZI (avoids any prior TIFF-save round-trip) ---
reference_czifile = CziFile(REFERENCE_CZI)
reference_img = blank_background(read_czi_rgb(reference_czifile, scale_factor))
normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
normalizer.fit(reference_img)

#--- normalize a handful of real CZIs and save for manual QuPath inspection ---
for czi_name in czi_files:
    czi_path = os.path.join(czi_dirpath, czi_name)
    print(f"Reading {czi_path}")
    czifile = CziFile(czi_path)
    img = blank_background(read_czi_rgb(czifile, scale_factor))

    # pixel size scales with the downsample: fewer pixels covering the same physical area
    out_pixel_size_um = get_pixel_size_um(czifile) / scale_factor

    norm_img, H, E = normalizer.normalize(I=img, stains=True)
    norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)

    out_name = os.path.splitext(czi_name)[0] + "_normalized.ome.tif"
    out_path = os.path.join(TEST_OUTPUT_DIR, out_name)
    tifffile.imwrite(
        out_path, norm_img,
        photometric="rgb", bigtiff=True, ome=True,
        resolution=(1e4 / out_pixel_size_um, 1e4 / out_pixel_size_um),
        resolutionunit="CENTIMETER",
        metadata={
            "PhysicalSizeX": out_pixel_size_um, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": out_pixel_size_um, "PhysicalSizeYUnit": "µm",
        },
    )
    print(f"Saved {out_path} (pixel size {out_pixel_size_um:.3f} um)")

print("Done. Open these in QuPath and check Image > Properties for pixel size == expected value.")
