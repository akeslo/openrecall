import sqlite3
from collections import namedtuple
import logging
import numpy as np
from typing import Any, List, Optional, Tuple

from openrecall.config import db_path

# Configure logging
logger = logging.getLogger(__name__)

# Define the structure of a database entry using namedtuple
Entry = namedtuple("Entry", ["id", "app", "title", "text", "timestamp", "embedding"])


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.

    The table schema includes columns for an auto-incrementing ID, application name,
    window title, extracted text, timestamp, and text embedding.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       app TEXT,
                       title TEXT,
                       text TEXT,
                       timestamp INTEGER UNIQUE,
                       embedding BLOB
                   )"""
            )
            # Add index on timestamp for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error during table creation: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute("SELECT id, app, title, text, timestamp, embedding FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array.
                # A NULL or truncated blob raises TypeError/ValueError, neither
                # of which is a sqlite3.Error — without this guard one bad row
                # aborts the whole fetch and 500s the search page.
                raw_embedding = row["embedding"]
                if raw_embedding is None:
                    logger.warning(
                        f"Skipping entry {row['id']}: embedding is NULL."
                    )
                    continue
                try:
                    embedding = np.frombuffer(raw_embedding, dtype=np.float32)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping entry {row['id']}: unreadable embedding blob ({e})."
                    )
                    continue
                if embedding.size == 0:
                    logger.warning(
                        f"Skipping entry {row['id']}: embedding is empty."
                    )
                    continue
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        timestamp=row["timestamp"],
                        embedding=embedding,
                    )
                )
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[int]:
    """
    Retrieves all timestamps from the database, ordered descending.

    Returns:
        List[int]: A list of all timestamps.
                   Returns an empty list if the table is empty or an error occurs.
    """
    timestamps: List[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            timestamps = [result[0] for result in results]
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching timestamps: {e}")
    return timestamps


def insert_entry(
    text: str, timestamp: int, embedding: np.ndarray, app: str, title: str
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entries (text, timestamp, embedding, app, title)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp) DO NOTHING""", # Avoid duplicates based on timestamp
                (text, timestamp, embedding_bytes, app, title),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        logger.error(f"Database error during insertion: {e}")
    return last_row_id


def delete_entry(entry_id: int) -> Optional[int]:
    """
    Deletes a single entry from the database by id.

    Args:
        entry_id (int): The id of the entry to delete.

    Returns:
        Optional[int]: The deleted row's timestamp if a row was deleted,
                       or None if no matching row existed or an error
                       occurred. The caller uses the timestamp to also
                       remove the entry's screenshot file(s) from disk —
                       this function only touches the database.
    """
    deleted_timestamp: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()
            if cursor.rowcount > 0:
                deleted_timestamp = row[0]
    except sqlite3.Error as e:
        logger.error(f"Database error during deletion of entry {entry_id}: {e}")
    return deleted_timestamp
