import sys
from pathlib import Path
import pytest
import subprocess
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_data(request):
    # Get the directory where conftest.py is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to your setup script
    setup_script = os.path.join(test_dir, "setup_test_data.py")

    # Run the setup script
    subprocess.run(["python", setup_script], check=True)

    # Optionally, add teardown code
    def teardown():
        # Add any cleanup code here
        pass

    request.addfinalizer(teardown)
