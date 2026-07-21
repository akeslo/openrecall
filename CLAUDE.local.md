<!--
  claude-md-updater: This file is automatically maintained after each git commit.
  To prevent a section from being auto-edited, wrap it:
    <!-- protected -->
    ## Your Section
    content here...
    <!-- /protected -->
-->

# CLAUDE.local.md — openrecall

Fork of upstream `openrecall/openrecall` (`git@upstream`), pushed to `akeslo/openrecall` (`origin`). Screen-recall/search tool.

## Tech Stack

- Python, installed via `setup.py` (`pip install -e .`)
- Flask (web app), torch + sentence-transformers (embeddings/ML), python-doctr (OCR)
- `requirements.txt` is a pinned lockfile generated with `pip-tools` (`pip-compile`) against `setup.py`'s base `install_requires` — OS-specific extras (`windows`/`macos`/`linux`/`python-doctr`) are excluded from the lockfile and installed separately via `pip install -e ".[macos]"` etc.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned base deps
pip install -e .                  # OpenRecall itself + OS extras via setup.py
python3 -m openrecall.app         # run
```

Regenerate the lockfile after editing `setup.py`'s `install_requires`:
```bash
pip install pip-tools
pip-compile --output-file=requirements.txt setup.py
```

## Architecture

<!-- add key structural notes here -->

## Conventions

<!-- add project-specific conventions here -->
