# Configuration file for Sphinx documentation builder.
from pathlib import Path

from omnicloudmask.__version__ import __version__

project = "OmniCloudMask"
copyright = "2025, Nick Wright"
author = "Nick Wright"
version = __version__
release = __version__
html_extra_path = ["googlef4150bdab6a1d90a.html"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_show_copyright = False
html_static_path = []

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Copy files from repo root into docs at build time
_docs_dir = Path(__file__).parent
_repo_root = _docs_dir.parent

_files_to_copy = [
    ("CHANGELOG.md", "changelog.md", None),
    ("MODEL_CHANGELOG.md", "model-changelog.md", "# Model Changelog\n\n"),
]

for src_rel, dst_name, prefix in _files_to_copy:
    src = _repo_root / src_rel
    dst = _docs_dir / dst_name
    if src.exists():
        content = src.read_text()
        if prefix:
            content = prefix + content
        dst.write_text(content)
