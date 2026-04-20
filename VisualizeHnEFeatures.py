import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image

czi_ds_path = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/HnE/Block07/HnE_IMG_0021.tif'
mask_path = '/Users/jbonaventura/Desktop/Annotations/HnE_R23-055_H7_4a_annotations_Mask.tiff'
csv_path = '/Users/jbonaventura/Desktop/Annotations/R23-055_H7_4a_features.csv'
geojson_path = '/Users/jbonaventura/Desktop/Annotations/R23-055_H7_4a_tiles.geojson'
mask_ds = 20  # must match ds used in WorkingGeoJson.py

czi_img = np.array(Image.open(czi_ds_path))
mask = np.array(Image.open(mask_path))
df = pd.read_csv(csv_path)

# Load tile polygons from GeoJSON — keyed by tile_id
with open(geojson_path) as f:
    geojson = json.load(f)

tile_polygons = {}
for feature in geojson['features']:
    tile_id = int(feature['properties']['name'].split('_')[1])
    # GeoJSON coords are [col, row] in full CZI space — convert to mask space
    coords = np.array(feature['geometry']['coordinates'][0][:-1])  # drop closing point
    tile_polygons[tile_id] = coords / mask_ds  # now in [col, row] = [x, y] mask space

label_colors = {'Muscle': 'green', 'Necrotic Tissue': 'yellow', 'Immune Infiltration': 'blue'}

fig, ax = plt.subplots(figsize=(14, 14))
ax.imshow(czi_img)
ax.imshow(mask, alpha=0.3, cmap='tab10')

for _, row in df.iterrows():
    tile_id = int(row['tile_id'])
    if tile_id not in tile_polygons:
        continue
    color = label_colors.get(row['label'], 'yellow')
    poly = Polygon(tile_polygons[tile_id], closed=True,
                   linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(poly)

ax.legend(handles=[patches.Patch(color=v, label=k) for k, v in label_colors.items()])
plt.tight_layout()
plt.show()

# CSV_PATH = '/Users/jbonaventura/Desktop/tile_features50.csv'
# GEOJSON_PATH = '/Users/jbonaventura/Desktop/tester50.geojson'
# DOWNSAMPLE = 50
#
# # Features to visualize — edit this list to show whatever you want
# FEATURES = [
# 'mean_Nucleus_Eosin_OD_range',
# 'mean_Cell_Area',
# 'mean_Cell_Perimeter',
# 'mean_Cell_Circularity',
# 'mean_Cell_Max_caliper',
# 'mean_Cell_Min_caliper',
# ]
#
# # --- Load tile polygons from GeoJSON ---
# with open(GEOJSON_PATH) as f:
# geojson = json.load(f)
#
# polygons = {}  # tile_name -> (N, 2) array of (x, y) corners, downsampled
# for feature in geojson['features']:
# name = feature['properties'].get('name')
# coords = np.array(feature['geometry']['coordinates'][0][:-1])  # drop closing point
# polygons[name] = coords / DOWNSAMPLE
#
# # --- Load features from CSV ---
# df = pd.read_csv(CSV_PATH)
# df = df[df['tile_name'].isin(polygons)]
#
# # --- Plot ---
# ncols = 3
# nrows = int(np.ceil(len(FEATURES) / ncols))
# fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
# axes = axes.flatten()
#
# for ax, feature in zip(axes, FEATURES):
# vals = df[feature].values
# norm = mcolors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
# cmap = cm.viridis
#
# patches = []
# colors = []
# for _, row in df.iterrows():
# poly = polygons[row['tile_name']]
# patches.append(Polygon(poly, closed=True))
# colors.append(row[feature])
#
# collection = PatchCollection(patches, cmap=cmap, norm=norm)
# collection.set_array(np.array(colors))
# ax.add_collection(collection)
# plt.colorbar(collection, ax=ax)
#
# ax.set_title(feature.replace('mean_', '').replace('_', ' '))
# ax.set_aspect('equal')
# ax.autoscale()
# ax.invert_yaxis()  # image coordinates: y increases downward
#
# # Hide any unused subplots
# for ax in axes[len(FEATURES):]:
# ax.set_visible(False)
#
# plt.tight_layout()
# plt.show()
