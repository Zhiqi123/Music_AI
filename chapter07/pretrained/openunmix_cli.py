"""Open-Unmix CLI wrapper for Chapter 7.

The openunmix package ships ``openunmix.cli`` but does not expose a
``__main__`` entry point, so ``python -m openunmix.cli`` imports the module
without running inference. This thin wrapper calls ``separate()`` with the
standard command-line arguments.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from openunmix.cli import separate

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    separate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
