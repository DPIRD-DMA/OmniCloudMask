try:
    from ._version import __version__
except ImportError:
    # Source checkout without a build step. setuptools-scm writes _version.py when
    # the package is built or installed, including editable installs.
    __version__ = "0.0.0+unknown"
