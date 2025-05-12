"""
Pytest configuration file for shared fixtures.
"""
import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to the Python path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def data_dir():
    """Return the path to the data directory."""
    return os.path.join(Path(__file__).parent.parent, "data")


@pytest.fixture
def admin_data_path(data_dir):
    """Return the path to the admin_lvl1.json file."""
    return os.path.join(data_dir, "admin_lvl1.json")


@pytest.fixture
def config_path():
    """Return the path to the config.yaml file."""
    return os.path.join(Path(__file__).parent.parent, "config.yaml") 