#!/usr/bin/env python3
"""Compatibility wrapper for the response regeneration entrypoint."""

from __future__ import annotations

import asyncio

from script_multiendpoint import main


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
