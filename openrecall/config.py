import os
import sys
import argparse
import logging

parser = argparse.ArgumentParser(description="OpenRecall")

parser.add_argument(
    "--storage-path",
    default=None,
    help="Path to store the screenshots and database",
)

parser.add_argument(
    "--primary-monitor-only",
    action="store_true",
    help="Only record the primary monitor",
    default=False,
)


# parse_known_args(), not parse_args(): this module is imported by every other
# module in the package (app.py, screenshot.py) as a side effect of their own
# imports, so anything that imports openrecall.* — a notebook, a REPL, a
# script, a future test module that imports before tests/conftest.py's
# sys.argv patch runs — hands argparse its own unrelated argv. parse_args()
# raises SystemExit on the first flag it doesn't recognize (pytest's `-q`,
# Jupyter's `--f=<connection file>`, etc.); parse_known_args() ignores what it
# doesn't recognize and keeps --storage-path/--primary-monitor-only working
# for real CLI invocations.
args, _unknown_args = parser.parse_known_args()


def get_appdata_folder(app_name="openrecall"):
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise EnvironmentError("APPDATA environment variable is not set.")
        path = os.path.join(appdata, app_name)
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        path = os.path.join(home, "Library", "Application Support", app_name)
    else:
        home = os.path.expanduser("~")
        path = os.path.join(home, ".local", "share", app_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if args.storage_path:
    appdata_folder = args.storage_path
    screenshots_path = os.path.join(appdata_folder, "screenshots")
    db_path = os.path.join(appdata_folder, "recall.db")
else:
    appdata_folder = get_appdata_folder()
    db_path = os.path.join(appdata_folder, "recall.db")
    screenshots_path = os.path.join(appdata_folder, "screenshots")

if not os.path.exists(screenshots_path):
    try:
        os.makedirs(screenshots_path, exist_ok=True)
    except OSError as e:
        # Do not swallow this silently: without this directory every
        # image.save() in the recorder thread fails, and a bare `except: pass`
        # here turns a one-line permissions/disk problem into "recording just
        # stopped working and nothing said why".
        logging.getLogger(__name__).error(
            f"Could not create screenshots directory '{screenshots_path}': {e}"
        )
