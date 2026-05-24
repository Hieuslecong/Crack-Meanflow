import sys
from pathlib import Path

CRACKDIFF_ROOT = "/home/hieulc/avitech11/crack_diff/crackdiff"
MEANFLOW_ROOT = "/home/hieulc/avitech11/MeanFlow"
CRACKMEANFLOW_ROOT = "/home/hieulc/avitech11/crackmeanflow"


def ensure_paths() -> None:
    """Add project roots needed by legacy CrackDiff/MeanFlow imports."""
    roots = [CRACKMEANFLOW_ROOT, CRACKDIFF_ROOT, str(Path(CRACKDIFF_ROOT).parent), MEANFLOW_ROOT]
    for root in reversed(roots):
        if root not in sys.path:
            sys.path.insert(0, root)


ensure_paths()
