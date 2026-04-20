#import shapely.geometry
from aicspylibczi import CziFile
import geopandas as gpd
import rasterio
from geopandas.array import geometry_type_values
from shapely import Polygon
from shapely import LineString
from rasterio import features
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from shapely import affinity
import json
from PIL import Image
import os
import re
import pandas as pd
import math
import ast

ds=20

czipath='/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/R23-055/R23-055_HnE_5x/block07/HnE_R23-055_H7_7a_5X.czi'
geojsonpath='/Users/jbonaventura/Desktop/Annotations/HnE_R23-055_H7_7a_annotations.geojson'
czifile = CziFile(czipath)
bbox = czifile.get_mosaic_bounding_box()
czi_img = czifile.read_mosaic(C=0, scale_factor=1/ds, region=(bbox.x, bbox.y, bbox.w, bbox.h), background_color=(1,1,1))[0,:,:,:]
print(czi_img.shape)
# Define the desired shape of the output raster (rows, columns)-
out_shape = (czi_img.shape[0], czi_img.shape[1])


states = gpd.read_file(geojsonpath)
#print(states.head(0))
print(states[["classification","geometry"]])



geoms=[]
dsgeoms=[]
tisslist=[]
for i in range(states.shape[0]):
    tissclass = states.at[i,"classification"]['name']
    geom = states.at[i,"geometry"]
    Gtype=geom.geom_type
    if Gtype == 'MultiPolygon':
        for poly in geom.geoms:
            # Access exterior of each individual Polygon
            coords = list(poly.exterior.coords)
            dscoords = [tuple(round(val / ds) for val in x) for x in coords]
            newShape= Polygon(dscoords)
            tisslist.append([newShape, tissclass])
    elif Gtype == 'LineString':
        coords = list(geom.coords)
        dscoords = [tuple(round(val / ds) for val in x) for x in coords]
        newShape = LineString(dscoords)
        tisslist.append([newShape, tissclass])
    else:
        coords = list(geom.exterior.coords)
        dscoords = [tuple(round(val / ds) for val in x) for x in coords]
        newShape = Polygon(dscoords)
        tisslist.append([newShape, tissclass])

sortedTis=sorted(tisslist, key=lambda tiss: tiss[0].area, reverse=True)
print([sublist[1] for sublist in sortedTis])

tisslabels={'Necrosis':100,'Immune cells':200}
MultiClassMask=np.zeros(out_shape,dtype=np.uint8)
for k in range(len(sortedTis)):
    shape=sortedTis[k][0]
    tissue=sortedTis[k][1]
    print(tissue)
    print("Center (Centroid):", shape.centroid.wkt)

    print("Extent (minx, miny, maxx, maxy):", shape.bounds)

    print(shape.area)

    burned_raster = rasterio.features.rasterize(
    shapes=[shape],
    out_shape=out_shape,
    fill=0,
    all_touched=True, # optional: rasterize all pixels touched by the geometry
    dtype=np.uint8
    )

    print(tisslabels[tissue])
    MultiClassMask=np.where(burned_raster==1, tisslabels[tissue], MultiClassMask)


plt.imshow(czi_img)
plt.imshow(MultiClassMask, alpha=0.5)
plt.show()


# # Save Image as Tiff-

#Better renaming bit-
anotedir= os.path.dirname(geojsonpath)
maskfilename=os.path.splitext(os.path.basename(geojsonpath))[0]



new_file_path = os.path.join(anotedir, maskfilename+ '_Mask.tiff')
CImage = Image.fromarray(MultiClassMask.astype(np.uint8))
CImage.save(new_file_path, 'TIFF')
