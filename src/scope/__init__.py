# -*- coding: utf-8 -*-
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # for Python<3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("scope")
except PackageNotFoundError:
    # package is not installed
    from ._version import __version__  # type: ignore
