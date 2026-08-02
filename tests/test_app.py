"""Tests for the Flask routes in openrecall.app.

app.py is the module the two most recent route fixes landed in (search input
validation, blanket route exception handling) and was the only module in the
package with no test coverage. These tests pin the behaviour those fixes
introduced: a failing dependency degrades to an empty page plus a 500 rather
than an unhandled traceback, and a blank query short-circuits before any
embedding work happens.

Every collaborator is patched at the openrecall.app namespace, so no database,
model, or screenshot directory is touched.
"""

import numpy as np
import pytest
from unittest import mock

from openrecall import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_timeline_renders_timestamps(client):
    with mock.patch.object(app_module, "get_timestamps", return_value=[1700000000]):
        response = client.get("/")
    assert response.status_code == 200
    assert b"1700000000" in response.data


def test_timeline_with_no_entries_still_renders(client):
    with mock.patch.object(app_module, "get_timestamps", return_value=[]):
        response = client.get("/")
    assert response.status_code == 200


def test_timeline_returns_500_when_lookup_fails(client):
    with mock.patch.object(
        app_module, "get_timestamps", side_effect=RuntimeError("db down")
    ):
        response = client.get("/")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

def _entry(timestamp, embedding):
    return mock.Mock(embedding=embedding, timestamp=timestamp)


def test_search_without_query_returns_empty_results(client):
    with mock.patch.object(app_module, "get_all_entries") as get_all_entries:
        response = client.get("/search")
    assert response.status_code == 200
    # Short-circuits before touching the database at all.
    get_all_entries.assert_not_called()


def test_search_with_blank_query_returns_empty_results(client):
    with mock.patch.object(app_module, "get_all_entries") as get_all_entries:
        response = client.get("/search?q=%20%20")
    assert response.status_code == 200
    get_all_entries.assert_not_called()


def test_search_orders_entries_by_descending_similarity(client):
    entries = [
        _entry(111, np.array([1.0, 0.0], dtype=np.float32)),
        _entry(222, np.array([0.0, 1.0], dtype=np.float32)),
    ]
    with mock.patch.object(app_module, "get_all_entries", return_value=entries), \
            mock.patch.object(
                app_module,
                "get_embedding",
                return_value=np.array([0.0, 1.0], dtype=np.float32),
            ), \
            mock.patch.object(
                app_module, "cosine_similarity", side_effect=[0.1, 0.9]
            ):
        response = client.get("/search?q=hello")

    assert response.status_code == 200
    body = response.data.decode()
    # The better match (timestamp 222) must be rendered before the weaker one.
    assert body.index("222") < body.index("111")


def test_search_returns_500_when_embedding_fails(client):
    with mock.patch.object(app_module, "get_all_entries", return_value=[]), \
            mock.patch.object(
                app_module, "get_embedding", side_effect=RuntimeError("model gone")
            ):
        response = client.get("/search?q=hello")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /static/<filename>
# ---------------------------------------------------------------------------

def test_serve_image_returns_file(client, tmp_path):
    (tmp_path / "1700000000_0.webp").write_bytes(b"not-really-a-webp")
    with mock.patch.object(app_module, "screenshots_path", str(tmp_path)):
        response = client.get("/static/1700000000_0.webp")
    assert response.status_code == 200
    assert response.data == b"not-really-a-webp"


def test_serve_image_missing_file_returns_404(client, tmp_path):
    with mock.patch.object(app_module, "screenshots_path", str(tmp_path)):
        response = client.get("/static/does-not-exist.webp")
    assert response.status_code == 404


def test_serve_image_rejects_path_traversal(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    with mock.patch.object(app_module, "screenshots_path", str(screenshots)):
        response = client.get("/static/..%2Fsecret.txt")
    assert response.status_code == 404
    assert b"classified" not in response.data
