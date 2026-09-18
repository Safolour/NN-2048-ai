"""Pytest bootstrap for the M0 test suite.

Makes the ``src`` layout importable without installing the package, so that::

    python -m pytest

works from the repository root.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
