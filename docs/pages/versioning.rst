Versioning
===========

Philosophy
-----------
:code:`scope` generally follows `semantic versioning <https://semver.org/>`_
guidelines. In brief, this means that our version numbers follow a `AA.BB.CC`
convenion, with `AA` incrementing when a large, backward-incompatible changes
are introduced to the code base; `BB` incrementing when backward-compatible
functionality is added to a code; and `CC` incrementing whenever backward-compatible
bugs are fixed.

Additionally, suffixes (e.g. "-beta" or "-dev") may be added to the release tag,
signifying the degree to which developers are confident in production-ready
(i.e., bug-free) code.


In practice
------------
This only applies to developers with write access to the code base. All that
needs to be done is make a new release on GitHub---the ``pyproject.toml`` file will automatically
read this information and update the package accordingly.
