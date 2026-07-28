"""
Percentile helpers shared across the simulation package.

Two estimators are in use and they are *not* interchangeable:

``percentile``    linear interpolation between the two bracketing samples.
                  Used for the reported flow statistics (topic / subscriber
                  latency percentiles in the message-flow results).
``nearest_rank``  index-based selection, returning an actual observed sample.
                  Used for the runtime latency properties surfaced through the
                  API, which have always been reported this way.

They agree only when the requested quantile lands exactly on a sample. Keep the
call sites on the estimator they already use — switching one silently shifts
every reported latency number.
"""

from typing import List, Optional, Sequence


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated *q*-th percentile (0-100). None when there is no data."""
    if not values:
        return None
    s: List[float] = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Index-based *q*-th percentile (0-100), returning an observed sample. 0.0 when empty."""
    if not values:
        return 0.0
    s: List[float] = sorted(values)
    idx = int(len(s) * (q / 100.0))
    return s[min(idx, len(s) - 1)]
