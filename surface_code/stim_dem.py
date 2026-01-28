from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import stim


def detector_error_model(circuit: stim.Circuit, *, decompose: bool = True) -> stim.DetectorErrorModel:
    """Get detector error model; decompose=True is typically needed for MWPM decoding."""
    return circuit.detector_error_model(decompose_errors=decompose)


def detector_matchgraph(
    circuit: stim.Circuit,
    *,
    decompose: bool = True,
    kind: Literal["svg", "svg-html", "3d", "3d-html"] = "svg-html",
):
    """Return a diagram object for Jupyter display or file writing.

    kind:
      - 'svg':      SVG text
      - 'svg-html': HTML iframe wrapping SVG (best in notebooks)
      - '3d':       GLTF model text
      - '3d-html':  HTML page wrapping the 3D viewer

    Notes:
      * '3d-html' 依赖外部 JS CDN，部分环境会被禁用；此时推荐用 '3d' 导出 glTF 再外部打开。
    """
    dem = detector_error_model(circuit, decompose=decompose)
    type_map = {
        "svg": "matchgraph-svg",
        "svg-html": "matchgraph-svg-html",
        "3d": "matchgraph-3d",
        "3d-html": "matchgraph-3d-html",
    }
    return dem.diagram(type_map[kind])


def sample_detectors_and_observables(
    circuit: stim.Circuit,
    shots: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample detector events and logical observables.

    Returns:
      dets: (shots, num_detectors)
      obs:  (shots, num_observables)
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return dets, obs
