#!/usr/bin/env python
"""Run the opt-in local proxy dogfood harness from a source checkout."""

from __future__ import annotations

import sys

from hermes.plugins.model_router.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["dogfood", "proxy", *sys.argv[1:]]))
