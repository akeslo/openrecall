import os
from threading import Thread

import numpy as np
from flask import Flask, render_template, request, send_from_directory

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import create_db, get_all_entries, get_timestamps
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable

# Set environment variable early to prevent tokenizers warnings in multiprocessing contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the path to the templates directory
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable


@app.route("/")
def timeline():
    # connect to db
    timestamps = get_timestamps()
    return render_template("timeline.html", timestamps=timestamps)


@app.route("/search")
def search():
    q = request.args.get("q")

    # Validate input
    if q is None or (isinstance(q, str) and not q.strip()):
        return render_template("search_results.html", entries=[])

    try:
        entries = get_all_entries()
        embeddings = [entry.embedding for entry in entries]
        query_embedding = get_embedding(q)
        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        indices = np.argsort(similarities)[::-1]
        sorted_entries = [entries[i] for i in indices]

        return render_template("search_results.html", entries=sorted_entries)
    except Exception as e:
        # Log the error and return empty results on failure
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error during search for query '{q}': {e}")
        return render_template("search_results.html", entries=[]), 500


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
