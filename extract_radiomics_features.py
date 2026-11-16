import logging
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# PyRadiomics logs a routine "GLCM is symmetrical..." style notice per call --
# silenced so a real run's output (one call per tile per contrast) isn't
# spammed with it.
logging.getLogger('radiomics').setLevel(logging.ERROR)

RADIOMICS_FEATURE_CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']


def build_radiomics_extractor():
    """One shared PyRadiomics extractor, force2D=True (tiles here are always
    single 2D planes -- matches this pipeline's per-slice convention, never a
    real 3D volume). Built once and reused across every tile/contrast call
    rather than per-call, since feature-class configuration has real setup
    cost (see timing check).

    binCount=32 (fixed bin COUNT, not the default fixed bin WIDTH=25):
    PyRadiomics' default binWidth assumes CT-scale absolute intensity units.
    Our MR contrasts are each independently rescaled to [0,1] (see
    load_mr_volumes' 'Normalized' convention), so a range-agnostic binWidth
    of 25 collapses every tile into a single gray-level bin -- confirmed
    empirically (glcm_Contrast=0, glcm_Correlation=1 on every tile
    regardless of content). binCount adapts bin edges to each ROI's own
    observed min/max instead, the standard fix for MR/PET (relative,
    non-standardized intensities) per PyRadiomics' own docs."""
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True, binCount=32)
    extractor.disableAllFeatures()
    for feature_class in RADIOMICS_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def radiomics_features(extractor, arr2d, mask2d=None):
    """PyRadiomics feature dict for one 2D single-contrast tile. mask2d
    defaults to MR's own no-data convention (0 = invalid, confirmed elsewhere
    in this pipeline) -- pass an explicit mask instead to restrict extraction
    to a segmented structure (e.g. a Frangi vessel mask) rather than the
    whole tile. Returns {name: float}; diagnostics_* keys (PyRadiomics'
    own run metadata, not features) are dropped, and the redundant
    'original_' prefix (only relevant once derived-image filters like
    wavelet/LoG are in play, which they aren't here) is stripped."""
    if mask2d is None:
        mask2d = arr2d != 0
    img = sitk.GetImageFromArray(arr2d[np.newaxis, :, :].astype(np.float32))
    msk = sitk.GetImageFromArray(mask2d.astype(np.uint8)[np.newaxis, :, :])
    result = extractor.execute(img, msk)
    return {k.removeprefix('original_'): float(v) for k, v in result.items()
            if not k.startswith('diagnostics')}
