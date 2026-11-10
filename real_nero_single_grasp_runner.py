#!/usr/bin/env python3
"""Backward-compatible entry point for :mod:`real_nero_grasp`.

Existing shell commands and imports keep using this filename.  When imported,
the module is aliased to the canonical runtime module so legacy monkeypatches
and helper imports continue to operate on the actual function globals.
"""

import sys

from real_nero_grasp import runtime as _runtime


if __name__ == "__main__":
    raise SystemExit(_runtime.main())

sys.modules[__name__] = _runtime
