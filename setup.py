"""Thin shim so editable installs (`pip install -e .`) work on tooling that still expects a
setup.py. All real metadata lives in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
