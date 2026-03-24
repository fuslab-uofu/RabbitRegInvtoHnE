import numpy as np
import nibabel as nib
import pytest
from AssessReg import segment_from_seeds, flatten_field, make_seg_nifti


def make_arr():
    """Simple 5x5x5 array with intensities 0-124."""
    return np.arange(125, dtype=float).reshape(5, 5, 5)


def test_empty_seeds_returns_zeros():
    arr = make_arr()
    result = segment_from_seeds(arr, [])
    assert result.shape == arr.shape
    assert not np.any(result)


def test_output_shape_matches_input():
    arr = make_arr()
    result = segment_from_seeds(arr, [(2, 2, 2)])
    assert result.shape == arr.shape


def test_single_seed_selects_exact_intensity():
    arr = make_arr()
    seed = (2, 2, 2)
    val = arr[seed]  # 62.0
    result = segment_from_seeds(arr, [seed])
    # With one seed min==max, only that exact intensity should match
    assert result[seed]
    assert np.all(result == (arr == val))


def test_two_seeds_selects_range():
    arr = make_arr()
    lo_seed = (0, 0, 1)  # intensity 1
    hi_seed = (0, 0, 5)  # intensity 5 (linear index 5 → (0,1,0))
    # Recompute: arr[i,j,k] = i*25 + j*5 + k
    lo_seed = (0, 0, 1)  # 1
    hi_seed = (0, 1, 0)  # 5
    lo_val = arr[lo_seed]  # 1
    hi_val = arr[hi_seed]  # 5
    result = segment_from_seeds(arr, [lo_seed, hi_seed])
    expected = (arr >= lo_val) & (arr <= hi_val)
    np.testing.assert_array_equal(result, expected)


def test_range_is_inclusive_on_both_ends():
    arr = np.array([[[1.0, 5.0, 10.0]]])
    seeds = [(0, 0, 0), (0, 0, 2)]  # intensities 1.0 and 10.0
    result = segment_from_seeds(arr, seeds)
    # All three values (1, 5, 10) fall in [1, 10]
    assert np.all(result)


def test_voxels_outside_range_excluded():
    arr = np.array([[[0.0, 5.0, 10.0, 20.0]]])
    seeds = [(0, 0, 1), (0, 0, 2)]  # intensities 5.0 and 10.0
    result = segment_from_seeds(arr, seeds)
    assert not result[0, 0, 0]   # 0.0 below range
    assert result[0, 0, 1]       # 5.0 in range
    assert result[0, 0, 2]       # 10.0 in range
    assert not result[0, 0, 3]   # 20.0 above range


def test_output_dtype_is_bool():
    arr = make_arr()
    result = segment_from_seeds(arr, [(1, 1, 1)])
    assert result.dtype == bool


def test_in_range_but_disconnected_from_seeds_excluded():
    # Two blobs of value 10 separated by a gap (value 1, out of range [10,10])
    arr = np.array([[[10.0, 1.0, 10.0]]])
    seeds = [(0, 0, 0)]  # only seed the left blob
    result = segment_from_seeds(arr, seeds)
    assert result[0, 0, 0]   # seed blob kept
    assert not result[0, 0, 1]  # gap excluded (out of range)
    assert not result[0, 0, 2]  # right blob excluded (in range but disconnected)


def test_multiple_seed_components_both_kept():
    # Two blobs of value 8-10, separated by a gap (value 1, out of range)
    arr = np.array([[[10.0, 1.0, 8.0]]])
    seeds = [(0, 0, 0), (0, 0, 2)]  # one seed in each blob
    result = segment_from_seeds(arr, seeds)
    assert result[0, 0, 0]   # left blob kept
    assert not result[0, 0, 1]  # gap excluded
    assert result[0, 0, 2]   # right blob kept


# --- segment_from_seeds cut tests ---

def test_cut_no_cuts_unchanged():
    """Passing no cuts should behave identically to the original function."""
    arr = np.array([[[10.0, 1.0, 10.0]]])
    seeds = [(0, 0, 0)]
    assert np.array_equal(
        segment_from_seeds(arr, seeds),
        segment_from_seeds(arr, seeds, cuts=[]),
    )


def test_cut_severs_connection_drops_unseeded_component():
    """Zeroing the bridge between two blobs should drop the component with no seed."""
    # Three voxels in a row: blob-A (10) — bridge (10) — blob-B (10)
    arr = np.array([[[10.0, 10.0, 10.0]]])
    seeds = [(0, 0, 0)]              # only seed blob-A
    cuts  = [(0, 0, 1, 0)]           # cut the bridge, radius=0 (single voxel)
    result = segment_from_seeds(arr, seeds, cuts=cuts)
    assert result[0, 0, 0]           # blob-A kept (contains seed)
    assert not result[0, 0, 1]       # bridge removed by cut
    assert not result[0, 0, 2]       # blob-B dropped (disconnected, no seed)


def test_cut_keeps_component_when_both_sides_seeded():
    """If both components are seeded, severing the bridge still keeps both."""
    arr = np.array([[[10.0, 10.0, 10.0]]])
    seeds = [(0, 0, 0), (0, 0, 2)]
    cuts  = [(0, 0, 1, 0)]           # radius=0 so only bridge voxel removed
    result = segment_from_seeds(arr, seeds, cuts=cuts)
    assert result[0, 0, 0]
    assert not result[0, 0, 1]       # cut voxel removed
    assert result[0, 0, 2]


def test_cut_radius_removes_cube():
    """radius=1 should remove a 3x3x3 cube, not just the centre."""
    arr = np.ones((5, 5, 5)) * 10.0
    seeds = [(0, 0, 0)]
    cuts  = [(2, 2, 2, 1)]           # radius stored in the tuple
    result = segment_from_seeds(arr, seeds, cuts=cuts)
    # Entire 3x3x3 block around centre must be False
    assert not result[2, 2, 2]
    assert not result[1, 2, 2]
    assert not result[3, 2, 2]
    assert not result[2, 1, 2]
    assert not result[2, 3, 2]
    assert not result[2, 2, 1]
    assert not result[2, 2, 3]
    assert not result[1, 1, 1]       # corner of cube also removed


# --- segment_from_seeds hi_bound tests ---

def test_hi_bound_none_uses_max_seed_intensity():
    """Default (hi_bound=None) should use max seed intensity as upper threshold."""
    arr = np.array([[[1.0, 5.0, 10.0, 20.0]]])
    seeds = [(0, 0, 1)]   # intensity 5
    result = segment_from_seeds(arr, seeds)
    assert result[0, 0, 1]       # seed voxel in
    assert not result[0, 0, 2]   # 10 > max_seed(5), excluded
    assert not result[0, 0, 3]   # 20 > max_seed(5), excluded


def test_hi_bound_expands_upper_threshold():
    """hi_bound=20 should include voxels up to intensity 20."""
    arr = np.array([[[1.0, 5.0, 10.0, 20.0]]])
    seeds = [(0, 0, 1)]   # intensity 5
    result = segment_from_seeds(arr, seeds, hi_bound=20.0)
    assert result[0, 0, 1]   # seed voxel in
    assert result[0, 0, 2]   # 10 <= 20, now included
    assert result[0, 0, 3]   # 20 <= 20, now included


def test_hi_bound_still_respects_connectivity():
    """Voxels within hi_bound range but disconnected from seeds are still excluded."""
    # blob-A (5) — gap (1, below lo) — blob-B (10)
    arr = np.array([[[5.0, 1.0, 10.0]]])
    seeds = [(0, 0, 0)]    # seed blob-A only
    result = segment_from_seeds(arr, seeds, hi_bound=10.0)
    assert result[0, 0, 0]      # blob-A kept
    assert not result[0, 0, 1]  # gap (intensity 1 < lo=5) breaks connection
    assert not result[0, 0, 2]  # blob-B disconnected even though in range


# --- make_seg_nifti tests ---

def test_make_seg_nifti_dtype_is_uint8():
    mask = np.ones((5, 5, 5), dtype=bool)
    affine = np.eye(4)
    img = make_seg_nifti(mask, affine)
    assert img.get_fdata().dtype == np.float64   # nibabel always returns float64 from get_fdata
    assert np.array_equal(np.asanyarray(img.dataobj), mask.astype(np.uint8))


def test_make_seg_nifti_shape_preserved():
    mask = np.zeros((10, 8, 6), dtype=bool)
    img = make_seg_nifti(mask, np.eye(4))
    assert img.shape == (10, 8, 6)


def test_make_seg_nifti_affine_preserved():
    mask = np.ones((4, 4, 4), dtype=bool)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = make_seg_nifti(mask, affine)
    np.testing.assert_array_equal(img.affine, affine)


def test_make_seg_nifti_values_are_0_and_1():
    mask = np.array([[[True, False, True]]])
    img = make_seg_nifti(mask, np.eye(4))
    data = np.asanyarray(img.dataobj)
    assert set(data.flat) == {0, 1}


# --- flatten_field tests ---

def test_flatten_field_output_shape_matches():
    arr = np.random.rand(30, 30, 20).astype(np.float32) * 100
    result = flatten_field(arr)
    assert result.shape == arr.shape


def test_flatten_field_uniform_stays_uniform():
    """A uniform non-zero array should remain uniform after flattening."""
    arr = np.ones((30, 30, 20), dtype=np.float32) * 80.0
    result = flatten_field(arr)
    assert np.std(result) < 1e-3


def test_flatten_field_output_nonnegative():
    arr = (np.random.rand(30, 30, 20) * 100).astype(np.float32)
    result = flatten_field(arr)
    assert np.all(result >= 0)
