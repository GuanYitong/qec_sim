from __future__ import annotations

from typing import Optional

import numpy as np
import stim

from .stim_dem import detector_error_model, sample_detectors_and_observables

try:
    import pymatching
except Exception:
    pymatching = None


def estimate_logical_error_rate_mwpm(
    circuit: stim.Circuit,
    shots: int = 5000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """MWPM decode using PyMatching and estimate logical error rate per observable.

    Returns:
      rates: shape (num_observables,)
    """
    if pymatching is None:
        raise ImportError("未安装 pymatching：请 pip install pymatching")

    dem = detector_error_model(circuit, decompose=True)
    m = pymatching.Matching.from_detector_error_model(dem)

    dets, obs = sample_detectors_and_observables(circuit, shots=shots, seed=seed)

    # 兼容老版本 pymatching：没有 decode_batch 的情况
    if hasattr(m, "decode_batch"):
        pred = np.atleast_2d(m.decode_batch(dets))
    else:
        pred = np.vstack([np.atleast_1d(m.decode(d)) for d in dets])

    logical_err = (pred ^ obs).astype(np.uint8)
    return logical_err.mean(axis=0)
