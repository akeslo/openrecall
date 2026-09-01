import os
import re
import logging
from glob import glob
from threading import Thread

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import create_db, delete_entry, get_all_entries, get_timestamps
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable

# Set environment variable early to prevent tokenizers warnings in multiprocessing contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logger = logging.getLogger(__name__)

# Get the path to the templates directory
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable


@app.route("/")
def timeline():
    try:
        # connect to db
        timestamps = get_timestamps()
        return render_template("timeline.html", timestamps=timestamps)
    except Exception as e:
        logger.error(f"Error fetching timeline: {e}")
        return render_template("timeline.html", timestamps=[]), 500


@app.route("/search")
def search():
    q = request.args.get("q")

    # Validate input
    if q is None or (isinstance(q, str) and not q.strip()):
        return render_template("search_results.html", entries=[])

    try:
        query_embedding = get_embedding(q)

        # Only compare entries whose embedding has the same dimension as the
        # query. cosine_similarity() does a bare np.dot, so a single row left
        # over from a different embedding model raises ValueError here — and
        # the except below would turn that into an empty result page for
        # *every* query, silently and permanently, until that row is deleted.
        comparable = []
        for entry in get_all_entries():
            if entry.embedding.shape != query_embedding.shape:
                logger.warning(
                    f"Skipping entry {entry.id} in search: embedding has shape "
                    f"{entry.embedding.shape}, query has {query_embedding.shape}."
                )
                continue
            comparable.append(entry)

        similarities = [
            cosine_similarity(query_embedding, entry.embedding) for entry in comparable
        ]
        indices = np.argsort(similarities)[::-1]
        sorted_entries = [comparable[i] for i in indices]

        return render_template("search_results.html", entries=sorted_entries)
    except Exception as e:
        # Log the error and return empty results on failure
        logger.error(f"Error during search for query '{q}': {e}")
        return render_template("search_results.html", entries=[]), 500


@app.route("/entry/<int:entry_id>", methods=["DELETE"])
def delete_entry_route(entry_id):
    try:
        deleted = delete_entry(entry_id)
        if not deleted:
            return jsonify({"error": "Entry not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting entry {entry_id}: {e}")
        return jsonify({"error": "Deletion failed"}), 500


def _monitor_fallback(filename):
    """Find another monitor's capture for the same timestamp.

    The timeline and search templates always request `<timestamp>_0.webp`, but
    with `--primary-monitor-only` off, a capture may only exist for a secondary
    monitor (`<timestamp>_1.webp`, ...). Without this fallback those entries
    show up in the UI as permanently broken images.

    Returns the substitute filename, or None. The strict pattern match also
    keeps this from being a path-traversal helper.
    """
    match = re.fullmatch(r"(\d+)_\d+\.webp", filename)
    if not match:
        return None
    candidates = sorted(glob(os.path.join(screenshots_path, f"{match.group(1)}_*.webp")))
    return os.path.basename(candidates[0]) if candidates else None


@app.route("/static/<filename>")
def serve_image(filename):
    try:
        if not os.path.isfile(os.path.join(screenshots_path, filename)):
            alternative = _monitor_fallback(filename)
            if alternative:
                logger.info(f"Serving '{alternative}' in place of missing '{filename}'.")
                return send_from_directory(screenshots_path, alternative)
        return send_from_directory(screenshots_path, filename)
    except Exception as e:
        logger.error(f"Error serving image '{filename}': {e}")
        return {"error": "File not found"}, 404


if __name__ == "__main__":
    # Configure logging once, at the entry point. Library modules only ever
    # call getLogger(__name__); without this, every logger.info() below the
    # WARNING threshold of the root logger's last-resort handler is dropped.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    create_db()

    logger.info(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
