import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, label as ndi_label

# Flattens every disconnected-in-3D voxel ID's 3D fragments onto the BlockFace X,Y plane
# (Z is only ~4mm thick, so flattening loses little) to visually check whether large
# inter-fragment gaps (e.g. the 34.8mm outlier found by CheckDisconnectedFragmentGaps-style
# analysis) cluster near the image/tissue edges -- i.e. areas that may not be well covered
# by the registration -- or are scattered through the interior.

Rabbit = 'R24-058'
Block = 'Block06'
PAD_VOXELS = 20
rabbase = f'/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/{Rabbit}'
bf_label_path = os.path.join(rabbase, 'InVivo_MR/RegDataProc/Voxel_Maps',
                              f'voxel_tile_vol_unique_pad{PAD_VOXELS}_ComposedRegTo{Block}.nii.gz')
geoms_dir = os.path.join(rabbase, 'Analysis', Block)

bf_nib = nib.load(bf_label_path)
bf_labels = bf_nib.get_fdata().astype(np.int64)
X, Y, Z = bf_labels.shape
obj_slices = find_objects(bf_labels)

disc_ids = np.load(os.path.join(geoms_dir, f'disconnected_in_3d_ids_pad{PAD_VOXELS}.npy')).tolist()

# Tissue silhouette: any voxel with a real (nonzero) label anywhere through the thin
# z-stack, flattened to X,Y -- gives real anatomical/image-extent context to plot against.
tissue_silhouette = np.any(bf_labels > 0, axis=2)

fig, ax = plt.subplots(figsize=(14, 18))
ax.imshow(tissue_silhouette.T, cmap='Greys', alpha=0.3, origin='lower')

cmap = plt.get_cmap('hsv')
rng = np.random.default_rng(0)

for vid in disc_ids:
    sl = obj_slices[vid - 1]
    if sl is None:
        continue
    z_edge = sl[2].start == 0 or sl[2].stop == Z  # known z-boundary-truncation case -- flagged, not dropped
    local_mask = bf_labels[sl] == vid
    comp_labels, n_components = ndi_label(local_mask, structure=np.ones((3, 3, 3)))
    offset = np.array([s.start for s in sl])

    color = cmap(rng.random())
    centroids_xy = []
    for c in range(1, n_components + 1):
        coords = np.argwhere(comp_labels == c) + offset  # local crop -> full-volume voxel index
        centroid_xy = coords[:, :2].mean(axis=0)
        centroids_xy.append(centroid_xy)
        size = len(coords)
        ax.scatter(centroid_xy[0], centroid_xy[1], color=color, s=15 + 2 * np.sqrt(size),
                   edgecolors='black' if z_edge else 'none', linewidths=0.7, zorder=3)

    centroids_xy = np.array(centroids_xy)
    for i in range(len(centroids_xy)):
        for j in range(i + 1, len(centroids_xy)):
            ax.plot([centroids_xy[i, 0], centroids_xy[j, 0]],
                    [centroids_xy[i, 1], centroids_xy[j, 1]],
                    color=color, linewidth=0.6, alpha=0.6, zorder=2)

ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_aspect('equal')
ax.set_title(f'{len(disc_ids)} disconnected-in-3D voxel IDs, flattened over Z\n'
             'color = one voxel ID; dot = one 3D fragment centroid (size ~ voxel count);\n'
             'line = fragments of the same ID; black-outlined dot = ID touches z-boundary')
plt.tight_layout()

out_path = os.path.join(geoms_dir, f'disconnected_fragments_flatZ_pad{PAD_VOXELS}.png')
plt.savefig(out_path, dpi=150)
print(f'Saved {out_path}')
plt.show()
