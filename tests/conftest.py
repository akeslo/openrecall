import sys

# openrecall.config parses CLI args at import time via argparse.parse_args().
# Strip pytest's own argv (test paths, -q, etc.) before any test module can
# trigger that import, so config parsing doesn't choke on pytest's flags.
sys.argv = sys.argv[:1]
