"""Small helpers shared by the report-writing scripts."""

from __future__ import annotations

import math


def fmt(x, digits: int = 3) -> str:
    """Fixed-point number, or an em dash for missing values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"
