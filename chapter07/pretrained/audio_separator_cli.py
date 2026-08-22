"""audio-separator CLI wrapper for Chapter 7.

The audio-separator package ships ``audio_separator.utils.cli`` but does not
expose a ``__main__`` entry point, so ``python -m audio_separator.utils.cli``
imports the module without running inference. This thin wrapper calls
``main()`` with the standard command-line arguments.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from audio_separator.utils.cli import main as cli_main

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
