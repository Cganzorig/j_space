#!/usr/bin/env python3
"""Run the punctuation-watermark design with the shared gated CUDA runner."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACROSTIC = HERE.parent / "secondary_objective"
for path in (str(HERE), str(ACROSTIC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import watermark_experiment
import run_acrostic

run_acrostic.fx = watermark_experiment
run_acrostic.HERE = HERE

if __name__ == "__main__":
    run_acrostic.main()
