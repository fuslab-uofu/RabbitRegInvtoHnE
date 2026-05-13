import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2hed, hed2rgb


def mean_nonzero(arr, axis=None, **kwargs):
    masked = np.ma.masked_equal(arr, 0)
    result = np.ma.mean(masked, axis=axis)
    return result.filled(0) if isinstance(result, np.ma.MaskedArray) else result


def haralick_features(image_gray, ignore_zeros=True):
    # image_gray must be uint8
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=False)

    if ignore_zeros:
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0

    glcm_sum = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.where(glcm_sum > 0, glcm / glcm_sum, 0)

    asm = graycoprops(glcm, 'ASM').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = np.sqrt(asm)

    p = glcm.mean(axis=(2, 3))
    i, j = np.mgrid[0:256, 0:256]
    mu_i = (i * p).sum()

    variance = ((i - mu_i) ** 2 * p).sum()
    dissimilarity = (np.abs(i - j) * p).sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_prob = p.max()

    p_sum = np.zeros(2 * 256)
    for k in range(2 * 256):
        mask = (i + j) == k
        p_sum[k] = p[mask].sum()
    sum_avg = np.sum(np.arange(2 * 256) * p_sum)
    sum_entropy = -np.sum(p_sum * np.log(p_sum + 1e-10))
    sum_var = np.sum((np.arange(2 * 256) - sum_entropy) ** 2 * p_sum)

    p_diff = np.zeros(256)
    for k in range(256):
        mask = np.abs(i - j) == k
        p_diff[k] = p[mask].sum()
    diff_var = np.sum(np.arange(256) ** 2 * p_diff) - np.sum(np.arange(256) * p_diff) ** 2
    diff_entropy = -np.sum(p_diff * np.log(p_diff + 1e-10))

    return np.array([asm, contrast, correlation, variance, homogeneity,
                     sum_avg, sum_var, sum_entropy, entropy,
                     diff_var, diff_entropy, energy, dissimilarity, max_prob])


def extract_tile_features(czi_patch, tformed_quad, mask, mask_labels):
    """Compute H&E color stats and haralick features for a tile.
    Returns a feature dict, or None if the tile should be skipped (ambiguous mask label)."""
    from FeatureCrossAnalysis import label_tile_from_mask

    haralick_names = ['asm', 'contrast', 'correlation', 'variance', 'homogeneity',
                      'sum_avg', 'sum_var', 'sum_entropy', 'entropy',
                      'diff_var', 'diff_entropy', 'energy', 'dissimilarity', 'max_prob']

    patch_f = czi_patch.astype(float)
    patch_f[patch_f == 0] = np.nan
    flat = patch_f.reshape(-1, 3)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)

    features = {
        'mean_R': means[0], 'mean_G': means[1], 'mean_B': means[2],
        'std_R': stds[0], 'std_G': stds[1], 'std_B': stds[2],
    }

    redchan = czi_patch[:, :, 0].astype(np.uint8)
    features.update({f'red_{n}': v for n, v in zip(haralick_names, haralick_features(redchan))})
    bluechan = czi_patch[:, :, 2].astype(np.uint8)
    features.update({f'blue_{n}': v for n, v in zip(haralick_names, haralick_features(bluechan))})

    if mask is not None:
        label = label_tile_from_mask(mask, tformed_quad, mask_labels)
        if label is None:
            return None
        features['label'] = label

    # This is not built to go here yet placeholder code will need to be reworked a bit for functionality-
    # ds_list=[]
    # for r in range(5):
    # #Playing with downsampling->
    # r=r+1
    # block_size = (50*r, 50*r, 1)
    # downsampled_array_avg = block_reduce(czi_patch, block_size=block_size, func=mean_nonzero).astype(np.uint8)
    # ds_list.append(downsampled_array_avg)
    # #Mean intensity of all three color channels->
    # nancopy=czi_patch.astype(float)
    # nancopy[nancopy==0]=np.nan
    # HnEFeatures[tile, 2:5]= np.nanmean(np.nanmean(nancopy, axis=0), axis=0)

    # Texture features-
    # 1st convert patch to greyscale->
    # Apply to red and green color channels seperately->
    # redchan= czi_patch[:,:,0].astype(np.uint8)
    # textfeatures_red = haralick_features(redchan)
    # HnEFeatures[tile, 5:19] = textfeatures_red
    #
    # bluechan= czi_patch[:,:,2].astype(np.uint8)
    # textfeatures_blue = haralick_features(bluechan)
    # HnEFeatures[tile, 19:33] = textfeatures_blue

    # for s in range(5):
    # for s in range(5):
    # start= 33+ s*28
    # redchan = ds_list[s][:, :, 0].astype(np.uint8)
    # textfeatures_red = haralick_features(redchan)
    # HnEFeatures[i, start:start+14] = textfeatures_red
    # bluechan = ds_list[s][:, :, 2].astype(np.uint8)
    # textfeatures_blue = haralick_features(bluechan)
    # HnEFeatures[i, start+14:start+28] = textfeatures_blue

    # MR features- Likewise not flushed out at all!
    # dy, dx = np.gradient(MR_Tile)
    # MRFeatures[tile, 2] = np.mean(MR_Tile)
    # MRFeatures[tile, 3] = np.std(MR_Tile)
    # MRFeatures[tile, 4] = np.mean(dx)
    # MRFeatures[tile, 5] = np.mean(dy)
    #print(MRFeatures[tile, :6])
    return features


def extract_features(patch):
    """
    Extract HnE features from a masked patch.

    Parameters
    ----------
    patch : np.ndarray (H, W, 3) RGB uint8, zeros outside polygon

    Returns
    -------
    dict of {feature_name: value}
    """
    haralick_names = ['asm', 'contrast', 'correlation', 'variance', 'homogeneity',
                      'sum_avg', 'sum_var', 'sum_entropy', 'entropy',
                      'diff_var', 'diff_entropy', 'energy', 'dissimilarity', 'max_prob']

    ihc_hed = rgb2hed(patch)

    h_channel = ihc_hed[:, :, 0]
    e_channel = ihc_hed[:, :, 1]

    features = {
        'h_mean': np.mean(h_channel),
        'h_std':  np.std(h_channel),
        'e_mean': np.mean(e_channel),
        'e_std':  np.std(e_channel),
    }

    # Rescale float HED channels to uint8 for haralick
    def to_uint8(ch):
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            return ((ch - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        return np.zeros_like(ch, dtype=np.uint8)

    features.update({f'h_{n}': v for n, v in zip(haralick_names, haralick_features(to_uint8(h_channel)))})
    features.update({f'e_{n}': v for n, v in zip(haralick_names, haralick_features(to_uint8(e_channel)))})

    return features
