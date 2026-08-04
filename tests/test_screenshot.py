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


# ---------------------------------------------------------------------------
# record_screenshots_thread resilience
# ---------------------------------------------------------------------------

def test_record_loop_survives_a_capture_failure():
    """One bad iteration must not kill the recording thread.

    The loop runs unsupervised on a background thread; before it caught its own
    exceptions, a single failure (display change, disk error, OCR blowup) ended
    recording permanently while the web app carried on serving stale data.
    """
    calls = {"n": 0}

    def flaky_take_screenshots():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("display disconnected")
        if calls["n"] >= 4:
            raise KeyboardInterrupt  # break out of the infinite loop
        return [np.zeros((4, 4, 3), dtype=np.uint8)]

    with mock.patch.object(screenshot, "take_screenshots", flaky_take_screenshots), \
         mock.patch.object(screenshot, "is_user_active", return_value=True), \
         mock.patch.object(screenshot, "time") as fake_time:
        fake_time.sleep.return_value = None
        fake_time.time.return_value = 1700000000
        with pytest.raises(KeyboardInterrupt):
            screenshot.record_screenshots_thread()

    # It kept iterating past the failure on call 2 rather than dying there.
    assert calls["n"] >= 4


def test_record_loop_survives_initial_capture_failure():
    def always_fails():
        raise RuntimeError("no display")

    with mock.patch.object(screenshot, "take_screenshots", always_fails), \
         mock.patch.object(screenshot, "is_user_active", side_effect=KeyboardInterrupt), \
         mock.patch.object(screenshot, "time") as fake_time:
        fake_time.sleep.return_value = None
        with pytest.raises(KeyboardInterrupt):
            screenshot.record_screenshots_thread()


# ---------------------------------------------------------------------------
# _insert_with_free_timestamp
# ---------------------------------------------------------------------------

def test_insert_with_free_timestamp_uses_first_second_when_free():
    with mock.patch("openrecall.screenshot.insert_entry", return_value=1) as insert:
        result = screenshot._insert_with_free_timestamp(
            "text", 1000, np.zeros(3), "App", "Title"
        )
    assert result == 1000
    assert insert.call_count == 1


def test_insert_with_free_timestamp_steps_past_a_collision():
    # Two monitors changing in the same second collide on the UNIQUE timestamp
    # column; insert_entry returns None for the loser, which must not be lost.
    with mock.patch(
        "openrecall.screenshot.insert_entry", side_effect=[None, 7]
    ) as insert:
        result = screenshot._insert_with_free_timestamp(
            "text", 1000, np.zeros(3), "App", "Title"
        )
    assert result == 1001
    assert insert.call_count == 2


def test_insert_with_free_timestamp_gives_up_after_max_attempts():
    with mock.patch("openrecall.screenshot.insert_entry", return_value=None) as insert:
        result = screenshot._insert_with_free_timestamp(
            "text", 1000, np.zeros(3), "App", "Title", max_attempts=3
        )
    assert result is None
    assert insert.call_count == 3
