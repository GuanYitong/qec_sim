from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

import networkx as nx

_RE_ERROR = re.compile(r"^\s*error\(([^)]+)\)\s+(.*)\s*$")
_RE_SHIFT = re.compile(r"^\s*shift_detectors\s+(-?\d+)\s*$")


def _edge_weight_from_p(p: float) -> float:
    # MWPM 通常用 log-likelihood 权重：w = log((1-p)/p)
    p = min(max(float(p), 1e-15), 1 - 1e-15)
    return math.log((1 - p) / p)


def build_graph_and_edge_table_from_dem_text(dem: Any) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """Parse str(DetectorErrorModel) into a graph + an explicit error-event table.

    Returns:
      G:
        nodes: detector indices (int) + boundary 'B'
        edges: representative edges, keeping the smallest weight if multiple events map to same (u,v)
        edge attrs: w, p, obs_mask
      edge_table:
        list of all error-events (each line 'error(p) ...') with:
          u, v, p, w, obs_mask
        Here v may be 'B' if the event has a single detector target.
    """
    G = nx.Graph()
    B = "B"
    G.add_node(B)

    edge_table: List[Dict[str, Any]] = []
    det_offset = 0

    for line in str(dem).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m_shift = _RE_SHIFT.match(line)
        if m_shift:
            det_offset += int(m_shift.group(1))
            continue

        m_err = _RE_ERROR.match(line)
        if not m_err:
            continue

        p = float(m_err.group(1))
        w = _edge_weight_from_p(p)
        rest = m_err.group(2)

        dets: List[int] = []
        obs_mask = 0
        for tok in rest.split():
            if tok.startswith("D"):
                dets.append(det_offset + int(tok[1:]))
            elif tok.startswith("L"):
                obs_mask ^= (1 << int(tok[1:]))

        if len(dets) == 2:
            u, v = dets
        elif len(dets) == 1:
            u, v = dets[0], B
        else:
            # len(dets)==0 是纯 logical 事件；不进入 matching 图
            continue

        edge_table.append(dict(u=u, v=v, p=p, w=w, obs_mask=obs_mask))

        # 图里只保留 (u,v) 的“代表边”（weight 最小那条）
        if G.has_edge(u, v):
            if w < G[u][v]["w"]:
                G[u][v].update(w=w, p=p, obs_mask=obs_mask)
        else:
            G.add_edge(u, v, w=w, p=p, obs_mask=obs_mask)

    return G, edge_table
