"""
Compatibility wrapper.

`entry.py` imports `MainAgent` from this module, but the actual implementation
currently lives in `mainagent可部署.py`.
"""

from mainagent可部署 import MainAgent  # type: ignore

__all__ = ["MainAgent"]

