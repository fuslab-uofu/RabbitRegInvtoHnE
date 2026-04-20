import json
import os
import numpy as np
import torch
from scipy.ndimage import map_coordinates


def landmark_paths_from_folder(folder):
    """Return (source_path, target_path) for a folder containing source.mrk.json and target.mrk.json."""
    src = os.path.join(folder, 'source.mrk.json')
    tgt = os.path.join(folder, 'target.mrk.json')
    if not os.path.exists(src):
        raise FileNotFoundError(f"source.mrk.json not found in {folder}")
    if not os.path.exists(tgt):
        raise FileNotFoundError(f"target.mrk.json not found in {folder}")
    return src, tgt


def load_landmarks(path):
    """Load control points from a LandMarker .mrk.json file.

    Returns a dict mapping landmark id (str) -> position as np.array([x, y, z]) in RAS mm.
    """
    with open(path) as f:
        data = json.load(f)

    points = {}
    for cp in data["markups"][0]["controlPoints"]:
        points[cp["id"]] = np.array(cp["position"])
    return points


def _load_pair_affines(source_path):
    """Read moving and fixed affines from a source .mrk.json file.

    Returns (moving_affine_4x4, fixed_affine_4x4) as numpy arrays.
    """
    with open(source_path) as f:
        data = json.load(f)
    pair = data["_landmarkerMeta"]["pair"]
    moving_affine = np.array(pair["moving"]["affine_4x4"])
    fixed_affine = np.array(pair["fixed"]["affine_4x4"])
    fixed_shape_zyx = pair["fixed"]["shape_zyx"]
    return moving_affine, fixed_affine, fixed_shape_zyx


def _ras_to_vox(positions_mm, affine_4x4):
    """Convert (N, 3) RAS mm positions to voxel (X, Y, Z) using the affine."""
    inv = np.linalg.inv(affine_4x4)
    ones = np.ones((len(positions_mm), 1))
    homog = np.hstack([positions_mm, ones])
    return (inv @ homog.T).T[:, :3]


def _vox_to_ras(vox, affine_4x4):
    """Convert (N, 3) voxel (X, Y, Z) to RAS mm using the affine."""
    ones = np.ones((len(vox), 1))
    homog = np.hstack([vox, ones])
    return (affine_4x4 @ homog.T).T[:, :3]


def apply_dfield_to_landmarks(positions_mm, source_path, dfield_path, n_iter=20):
    """Transform landmark positions from moving RAS mm space into fixed RAS mm space.

    The deformation field is an inverse warp (field[fixed_vox] = moving_vox).
    Fixed-point iteration is used to invert it for the sparse set of landmark points.

    positions_mm: (N, 3) array of positions in moving space (RAS mm)
    source_path:  path to source .mrk.json (used to read moving and fixed affines)
    dfield_path:  path to .pt deformation field
    n_iter:       number of fixed-point iterations (20 is plenty for moderate deformations)

    Returns (N, 3) array of transformed positions in fixed space (RAS mm).
    """
    moving_affine, fixed_affine, fixed_shape_zyx = _load_pair_affines(source_path)

    data = torch.load(dfield_path, map_location='cpu')
    field = data['data'].numpy()  # [3, X, Y, Z], 1-indexed moving voxel coords

    # Crop field to fixed image spatial dims (field may be padded)
    # shape_zyx stores values in nibabel (X, Y, Z) order despite its name
    X, Y, Z = fixed_shape_zyx
    field = field[:, :X, :Y, :Z]

    # Convert source positions from RAS mm -> moving voxel (X, Y, Z)
    p_moving = _ras_to_vox(positions_mm, moving_affine)  # (N, 3)

    # Fixed-point iteration: find p_fixed such that field(p_fixed) == p_moving
    # field[2]-1 = X_moving, field[1]-1 = Y_moving, field[0]-1 = Z_moving (0-indexed)
    p_fixed = p_moving.copy()
    for _ in range(n_iter):
        coords = [p_fixed[:, 0], p_fixed[:, 1], p_fixed[:, 2]]
        x_est = map_coordinates(field[2], coords, order=1) - 1
        y_est = map_coordinates(field[1], coords, order=1) - 1
        z_est = map_coordinates(field[0], coords, order=1) - 1
        p_moving_est = np.stack([x_est, y_est, z_est], axis=1)
        p_fixed = p_fixed + (p_moving - p_moving_est)

    return _vox_to_ras(p_fixed, fixed_affine)


def compute_tre(source_path, target_path, dfield_path=None):
    """Compute Target Registration Error between paired landmark files.

    Landmarks are matched by id. Positions are in RAS mm so Euclidean
    distance is directly in mm.

    If dfield_path is provided, source landmarks are transformed through the
    deformation field before computing error (for assessing post-registration TRE).

    Returns:
        results: list of dicts with keys 'id', 'source', 'target', 'error_mm'
        mean_tre: float
        std_tre: float
    """
    source = load_landmarks(source_path)
    target = load_landmarks(target_path)

    ids = sorted(set(source.keys()) & set(target.keys()))
    if not ids:
        raise ValueError("No matching landmark ids between source and target files.")

    missing = set(source.keys()) ^ set(target.keys())
    if missing:
        print(f"Warning: landmark ids present in only one file, skipping: {missing}")

    source_positions = np.array([source[i] for i in ids])

    if dfield_path is not None:
        source_positions = apply_dfield_to_landmarks(source_positions, source_path, dfield_path)

    results = []
    for idx, lid in enumerate(ids):
        error = float(np.linalg.norm(source_positions[idx] - target[lid]))
        results.append({
            "id": lid,
            "source": source_positions[idx],
            "target": target[lid],
            "error_mm": error,
        })

    errors = np.array([r["error_mm"] for r in results])
    return results, float(errors.mean()), float(errors.std())


def print_tre_report(source_path, target_path, dfield_path=None):
    results, mean_tre, std_tre = compute_tre(source_path, target_path, dfield_path)
    label = "post-dfield" if dfield_path else "direct"
    print(f"TRE report ({label})")
    print(f"{'ID':<6} {'Error (mm)':<12}")
    print("-" * 18)
    for r in results:
        print(f"{r['id']:<6} {r['error_mm']:.3f}")
    print("-" * 18)
    print(f"{'Mean':<6} {mean_tre:.3f}")
    print(f"{'Std':<6} {std_tre:.3f}")


def view_tre(source_vol_path, target_vol_path, source_mrk_path, target_mrk_path, dfield_path=None):
    """Launch VolumeViewer windows showing landmark alignment before (and optionally after) dfield.

    source_vol_path: path to the source (moving) NIfTI volume
    target_vol_path: path to the target (fixed) NIfTI volume
    source_mrk_path: path to source .mrk.json
    target_mrk_path: path to target .mrk.json
    dfield_path:     optional path to .pt deformation field — if given, opens a second window
                     showing landmarks after the dfield is applied
    """
    import sys
    import nibabel as nib
    from PyQt5.QtWidgets import QApplication
    from Viewer import VolumeViewer

    # Load volumes in canonical space to match LandMarker's convention
    src_vol = nib.as_closest_canonical(nib.load(source_vol_path)).get_fdata().astype(np.float32)
    tgt_vol = nib.as_closest_canonical(nib.load(target_vol_path)).get_fdata().astype(np.float32)
    src_vol = src_vol / src_vol.max() * 255
    tgt_vol = tgt_vol / tgt_vol.max() * 255

    moving_affine, fixed_affine, _ = _load_pair_affines(source_mrk_path)

    src_lm = load_landmarks(source_mrk_path)
    tgt_lm = load_landmarks(target_mrk_path)
    ids = sorted(set(src_lm.keys()) & set(tgt_lm.keys()))

    src_pos_mm = np.array([src_lm[i] for i in ids])
    tgt_pos_mm = np.array([tgt_lm[i] for i in ids])

    src_vox = np.round(_ras_to_vox(src_pos_mm, moving_affine)).astype(int)
    tgt_vox = np.round(_ras_to_vox(tgt_pos_mm, fixed_affine)).astype(int)

    pre_landmarks = [
        {'vox': src_vox, 'color': 'red',  'label': 'source (pre-dfield)', 'ids': ids},
        {'vox': tgt_vox, 'color': 'cyan', 'label': 'target',              'ids': ids},
    ]

    app = QApplication.instance() or QApplication(sys.argv)

    viewer_pre = VolumeViewer(src_vol, tgt_vol, title='Landmarks: pre-dfield', landmarks=pre_landmarks)
    viewer_pre.show()

    if dfield_path is not None:
        transformed_mm = apply_dfield_to_landmarks(src_pos_mm, source_mrk_path, dfield_path)
        transformed_vox = np.round(_ras_to_vox(transformed_mm, fixed_affine)).astype(int)
        post_landmarks = [
            {'vox': transformed_vox, 'color': 'red',  'label': 'source (post-dfield)', 'ids': ids},
            {'vox': tgt_vox,         'color': 'cyan', 'label': 'target',               'ids': ids},
        ]
        viewer_post = VolumeViewer(src_vol, tgt_vol, title='Landmarks: post-dfield', landmarks=post_landmarks)
        viewer_post.show()

    app.exec_()

LMFolder = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegAssessData"
src, tgt = landmark_paths_from_folder(LMFolder)
dfield = "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegTransforms/LMExVtoBlockFace.pt"

view_tre(
    source_vol_path="/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegDataProc/ExVivoBlock06ResampledToBlockFace06.nii.gz",
    target_vol_path="/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/BlockFace_RGB/Block06/greyscale_downsampled.nii.gz",
    source_mrk_path=src,
    target_mrk_path=tgt,
    dfield_path=dfield,
)



# Without dfield (just direct position comparison)
print_tre_report(src, tgt)

# After applying dfield (proper post-registration TRE)
print_tre_report(src, tgt, dfield_path=dfield)

# import torch
# import numpy as np
#
# data = torch.load("/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/ExVivo_MRBlocked/Block06/RegTransforms/LMExVtoBlockFace.pt", map_location='cpu')
# field = data['data'].numpy()
# print("field shape:", field.shape)
# print("field[0] range (should be ~0-475):", field[0].min(), field[0].max())
# print("field[1] range (should be ~0-638):", field[1].min(), field[1].max())
# print("field[2] range (should be ~0-71):", field[2].min(), field[2].max())
#
# # Check what field gives at landmark 1's moving voxel position (197, 274, 0)
# print("field at [197, 274, 0]:", field[:, 197, 274, 0])
# # Should be close to [1, 275, 198] (1-indexed equivalent of [0, 274, 197])


