import numpy as np
import matplotlib.pyplot as plt
from aicspylibczi import CziFile
from HnEFeatureExtraction import extract_features

# --- Configure these ---
CZI_PATH = '/Users/jbonaventura/Downloads/HnE_R23-055_H6_7a_5X.czi'
czifile = CziFile(CZI_PATH)
bbox = czifile.get_mosaic_bounding_box()
SCALE_FACTOR = 1  # 1 = full res, 1/20 = downsampled
# region: (x, y, w, h) in CZI mosaic coords — set to None to load the full image
REGION = (bbox.x + 12000, bbox.y + 12000, 4000, 2500)
# -----------------------



region = REGION if REGION is not None else (bbox.x, bbox.y, bbox.w, bbox.h)
patch = czifile.read_mosaic(C=0, scale_factor=SCALE_FACTOR, region=region)[0]
patch[:, :, [0, 2]] = patch[:, :, [2, 0]]  # BGR→RGB

# Full patch mask (no polygon — tests all pixels)
mask = np.full(patch.shape[:2], 255, dtype=np.uint8)

# plt.imshow(patch)
# plt.title('Loaded patch')
# plt.axis('off')
# plt.show()

features = extract_features(patch)
for k, v in features.items():
    print(f'{k}: {v:.4f}')
