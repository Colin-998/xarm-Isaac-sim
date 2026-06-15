"""Load cuRobo V2 inside the Isaac Sim 5.1 Python environment on Windows."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUROBO_SOURCE = ROOT / "vendor" / "curobo"
CUROBO_RUNTIME = ROOT / "vendor" / "curobo_runtime"


def configure_curobo_imports():
    """Prefer Isaac's numeric stack and isolated cuRobo runtime dependencies."""
    # Import these before adding the runtime directory so Isaac Sim keeps its
    # compatible NumPy, Torch, SciPy, YAML, and Trimesh versions.
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import torch  # noqa: F401
    import trimesh  # noqa: F401
    import yaml  # noqa: F401

    for path in (CUROBO_RUNTIME, CUROBO_SOURCE):
        if not path.exists():
            raise RuntimeError(f"Missing cuRobo dependency path: {path}")
        path_text = str(path)
        if path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)

