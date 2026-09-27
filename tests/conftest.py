import os
import sys

# Ensure matplotlib can import headlessly before any module under test does
# `import matplotlib; matplotlib.use('TkAgg')`.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
