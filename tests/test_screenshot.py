import numpy as np
import pytest
from unittest import mock

from openrecall import screenshot


# ---------------------------------------------------------------------------
# mean_structured_similarity_index
# ---------------------------------------------------------------------------

def test_mssim_identical_images_is_one():
    img = np.random.randint(0, 255, (10, 10, 3)).astype(np.float64)
    result = screenshot.mean_structured_similarity_index(img, img)
    assert result == pytest.approx(1.0)


def test_mssim_different_images_less_than_one():
    img1 = np.zeros((10, 10, 3))
    img2 = np.full((10, 10, 3), 255.0)
    result = screenshot.mean_structured_similarity_index(img1, img2)
    assert result < 1.0


def test_mssim_is_symmetric():
    img1 = np.random.randint(0, 255, (10, 10, 3)).astype(np.float64)
    img2 = np.random.randint(0, 255, (10, 10, 3)).astype(np.float64)
    result_ab = screenshot.mean_structured_similarity_index(img1, img2)
    result_ba = screenshot.mean_structured_similarity_index(img2, img1)
    assert result_ab == pytest.approx(result_ba)


# ---------------------------------------------------------------------------
# is_similar
# ---------------------------------------------------------------------------

def test_is_similar_true_for_identical_images():
    img = np.random.randint(0, 255, (10, 10, 3)).astype(np.float64)
    assert bool(screenshot.is_similar(img, img)) is True


def test_is_similar_false_for_dissimilar_images():
    img1 = np.zeros((10, 10, 3))
    img2 = np.full((10, 10, 3), 255.0)
    assert bool(screenshot.is_similar(img1, img2, similarity_threshold=0.9)) is False


def test_is_similar_respects_custom_threshold():
    with mock.patch(
        "openrecall.screenshot.mean_structured_similarity_index", return_value=0.5
    ):
        img = np.zeros((2, 2, 3))
        assert screenshot.is_similar(img, img, similarity_threshold=0.4) is True
        assert screenshot.is_similar(img, img, similarity_threshold=0.6) is False


# ---------------------------------------------------------------------------
# take_screenshots
# ---------------------------------------------------------------------------

def _make_fake_sct(num_monitors_incl_all=3):
    """Builds a fake mss.mss() context manager with `num_monitors_incl_all`
    entries in .monitors (index 0 is the "all monitors" combined view)."""
    fake_sct = mock.MagicMock()
    fake_sct.monitors = [{"index": i} for i in range(num_monitors_incl_all)]
    fake_frame = mock.MagicMock()
    # Simulate a BGRA screenshot buffer via numpy indexing support.
    fake_frame_array = np.zeros((5, 5, 4), dtype=np.uint8)
    fake_sct.grab.return_value = fake_frame

    fake_cm = mock.MagicMock()
    fake_cm.__enter__.return_value = fake_sct
    fake_cm.__exit__.return_value = False
    return fake_cm, fake_sct, fake_frame_array


def test_take_screenshots_all_monitors():
    fake_cm, fake_sct, fake_frame_array = _make_fake_sct(num_monitors_incl_all=3)

    with mock.patch("openrecall.screenshot.mss.mss", return_value=fake_cm):
        with mock.patch("openrecall.screenshot.np.array", return_value=fake_frame_array):
            with mock.patch("openrecall.screenshot.args") as fake_args:
                fake_args.primary_monitor_only = False
                result = screenshot.take_screenshots()

    # monitors 1 and 2 captured (index 0 is the "all monitors" entry, skipped)
    assert len(result) == 2
    assert fake_sct.grab.call_count == 2
    for shot in result:
        assert shot.shape == (5, 5, 3)


def test_take_screenshots_primary_monitor_only():
    fake_cm, fake_sct, fake_frame_array = _make_fake_sct(num_monitors_incl_all=3)

    with mock.patch("openrecall.screenshot.mss.mss", return_value=fake_cm):
        with mock.patch("openrecall.screenshot.np.array", return_value=fake_frame_array):
            with mock.patch("openrecall.screenshot.args") as fake_args:
                fake_args.primary_monitor_only = True
                result = screenshot.take_screenshots()

    assert len(result) == 1
    assert fake_sct.grab.call_count == 1
    fake_sct.grab.assert_called_once_with({"index": 1})


def test_take_screenshots_skips_out_of_bounds_index():
    fake_cm, fake_sct, fake_frame_array = _make_fake_sct(num_monitors_incl_all=1)

    with mock.patch("openrecall.screenshot.mss.mss", return_value=fake_cm):
        with mock.patch("openrecall.screenshot.np.array", return_value=fake_frame_array):
            with mock.patch("openrecall.screenshot.args") as fake_args:
                # Only one monitor entry (the "all monitors" combined view),
                # so primary_monitor_only=True has no valid index 1 to grab.
                fake_args.primary_monitor_only = True
                result = screenshot.take_screenshots()

    assert result == []
    fake_sct.grab.assert_not_called()
