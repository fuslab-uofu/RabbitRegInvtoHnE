

import numpy as np
import torch
import timm

MODEL = "hf_hub:Snarcy/RadioDino-b16"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


_LOADED = {}        


def load_model(name=MODEL, device=None):

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (name, str(device))
    if key in _LOADED:
        return _LOADED[key]
    model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    model.eval().to(device)
    cfg = getattr(model, "pretrained_cfg", None) or {}
    info = dict(
        name=name,
        dim=int(model.num_features),
        prefix=int(model.num_prefix_tokens),
        patch=int(model.patch_embed.patch_size[0]),
        mean=tuple(cfg.get("mean", IMAGENET_MEAN)),
        std=tuple(cfg.get("std", IMAGENET_STD)),
        device=device,
    )
    _LOADED[key] = (model, info)
    return model, info


def _prepare(tile, patch):
    a = np.asarray(tile)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    assert a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3), (
        f"tile must be (H,W) or (H,W,3), got {a.shape}."
    )
    h, w = a.shape[:2]
    assert h % patch == 0 and w % patch == 0, (
        f"tile is {h}x{w}; both sides must be multiples of the patch size {patch} so that "
        f"tokens line up with pixels. Crop to {(h // patch) * patch}x{(w // patch) * patch}."
    )

    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
        assert np.isfinite(a).all(), (
            "tile contains NaN or inf. Replace them before embedding, and keep a mask if "
            "you need to exclude those patches downstream."
        )
        assert a.min() >= 0.0 and a.max() <= 1.0, (
            f"tile has range [{a.min():.4g}, {a.max():.4g}], not [0,1]. Normalize the whole "
            "volume first, with ONE fixed (lo, hi) per contrast — not per tile, which would "
            "give every tile a different intensity scale."
        )

    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def embed_tile(tiles, name=MODEL, device=None, batch_size=16):
    """One tile -> grid (gh,gw,D), cls (D,), mean (D,).
    A list of equally-shaped tiles -> the same with a leading batch axis."""
    single = not isinstance(tiles, (list, tuple))
    if single:
        tiles = [tiles]

    model, info = load_model(name, device)
    patch, prefix = info["patch"], info["prefix"]
    arrays = [_prepare(t, patch) for t in tiles]
    assert len({a.shape for a in arrays}) == 1, (
        f"all tiles in one call must have the same shape, got {sorted({a.shape for a in arrays})}. "
        "Call embed_tile once per shape."
    )

    mean_t = torch.tensor(info["mean"]).view(1, 3, 1, 1)
    std_t = torch.tensor(info["std"]).view(1, 3, 1, 1)
    grids, clss, means = [], [], []

    for i in range(0, len(arrays), batch_size):
        x = torch.from_numpy(np.stack(arrays[i:i + batch_size]))
        x = ((x - mean_t) / std_t).to(info["device"])
        out = model.forward_features(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        b, n, d = out.shape
        gh, gw = x.shape[2] // patch, x.shape[3] // patch

        tok = out[:, prefix:]
        grids.append(tok.reshape(b, gh, gw, d).detach().float().cpu().numpy())
        means.append(tok.mean(dim=1).detach().float().cpu().numpy())
        clss.append((out[:, 0] if prefix else tok.mean(dim=1)).detach().float().cpu().numpy())

    out = dict(grid=np.concatenate(grids), cls=np.concatenate(clss),
               mean=np.concatenate(means), patch=patch, dim=info["dim"])
    if single:
        out.update(grid=out["grid"][0], cls=out["cls"][0], mean=out["mean"][0])
    return out


def token_at(grid, y, x, patch=16):
    i, j = y // patch, x // patch
    assert 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1], (
        f"pixel ({y},{x}) -> token ({i},{j}) is outside the {grid.shape[:2]} grid."
    )
    return grid[i, j]


def patch_pixel_center(tile_cx, tile_cy, tile_size, patch, i, j):
    """Pixel (x,y) of the single 'central' pixel representing patch (i,j) in a
    (gh,gw) token grid cropped from a tile centered at (tile_cx,tile_cy) --
    the inverse of token_at's pixel->patch lookup. Patches are non-overlapping
    and start at the tile's own top-left corner (tile_cx-half, matching
    ExtractMRFeatures.extract_tile's own convention), so this lines up exactly
    with how the tile was cropped. A patch has no single center pixel when
    `patch` is even (its true center falls between two pixels) -- convention:
    round down to the lower one, via (patch-1)//2."""
    half = tile_size // 2
    tile_x0, tile_y0 = tile_cx - half, tile_cy - half
    offset = (patch - 1) // 2
    return tile_x0 + i * patch + offset, tile_y0 + j * patch + offset
