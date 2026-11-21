import sys
from pathlib import Path

# Make the repository root importable so tests can import the original SCARCE modules
# (utils_algo.py etc.) and the scarda package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
