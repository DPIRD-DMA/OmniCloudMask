import subprocess
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

requires_fastai = pytest.mark.skipif(
    __import__("importlib").util.find_spec("fastai") is None,
    reason="fastai not installed (legacy models, pip install omnicloudmask[legacy])",
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_data(request: pytest.FixtureRequest) -> None:
    """Download the test scenes, but only if a selected test needs them. Tests
    that read the scenes are marked slow, so `pytest -m "not slow"` runs without
    the large download."""
    if not any(item.get_closest_marker("slow") for item in request.session.items):
        return
    setup_script = Path(__file__).parent / "setup_test_data.py"
    subprocess.run([sys.executable, str(setup_script)], check=True)
