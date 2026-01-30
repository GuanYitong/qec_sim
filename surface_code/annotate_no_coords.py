from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import stim

from .dem_graph import build_graph_and_edge_table_from_dem_text

Node = Union[int, str]  # detector index or 'B'


def _mwpm_pairs_with_boundary(
    G: nx.Graph, fired: List[int]
) -> Tuple[List[Tuple[Node, Node]], Dict[Tuple[Node, Node], List[Node]]]:
    """Explain-only MWPM (NetworkX), allow matching to boundary via boundary copies."""
    B = "B"
    fired = list(fired)
    if len(fired) == 0:
        return [], {}

    # precompute shortest paths in base graph
    dists = {}
    paths = {}
    for s in fired:
        lengths, pths = nx.single_source_dijkstra(G, s, weight="w")
        dists[s] = lengths
        paths[s] = pths

    # Build a complete graph for matching; allow multiple matches to boundary by duplicating it.
    k = len(fired)
    Bcopies = [("B", i) for i in range(k)]

    K = nx.Graph()
    for u in fired:
        K.add_node(u)
    for b in Bcopies:
        K.add_node(b)

    # fired-fired edges
    for i, u in enumerate(fired):
        for v in fired[i + 1 :]:
            K.add_edge(u, v, weight=dists[u].get(v, 1e9))

    # fired-boundary edges
    for u in fired:
        w = dists[u].get(B, 1e9)
        for b in Bcopies:
            K.add_edge(u, b, weight=w)

    # boundary-boundary edges (0 cost)
    for i in range(k):
        for j in range(i + 1, k):
            K.add_edge(Bcopies[i], Bcopies[j], weight=0.0)

    M = nx.algorithms.matching.min_weight_matching(K, weight="weight") # type: ignore

    pairs: List[Tuple[Node, Node]] = []
    pair_paths: Dict[Tuple[Node, Node], List[Node]] = {}

    used = set()
    for a, b in M:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)

        if a in fired and b in fired:
            u, v = a, b
            pairs.append((u, v))
            pair_paths[(u, v)] = paths[u].get(v) or paths[v].get(u) or [u, v]
        elif a in fired and isinstance(b, tuple) and b[0] == "B":
            u = a
            pairs.append((u, "B"))
            pair_paths[(u, "B")] = paths[u].get("B") or [u, "B"]
        elif b in fired and isinstance(a, tuple) and a[0] == "B":
            u = b
            pairs.append((u, "B"))
            pair_paths[(u, "B")] = paths[u].get("B") or [u, "B"]

    return pairs, pair_paths


def _plot_3d_graph_plotly(
    G: nx.Graph,
    *,
    pos: dict[Node, tuple[float, float, float]],
    fired: list[int],
    highlight_edges: list[tuple[int, int, str]],
    injected_edge: tuple[Node, Node] | None = None,
    title: str,
):
    B = "B"

    # ---- 1) 全图 edges（灰）----
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x1, y1, z1 = pos[u]
        x2, y2, z2 = pos[v]
        ex += [x1, x2, None]
        ey += [y1, y2, None]
        ez += [z1, z2, None]

    traces = [
        go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            line=dict(width=2),
            name="matchgraph",
            opacity=0.25,
        )
    ]

    # ---- 2) 全图节点（黑）----
    nx_all, ny_all, nz_all, txt_all = [], [], [], []
    for n in G.nodes():
        if n == B:
            continue
        if n not in pos:
            continue
        x, y, z = pos[n]
        nx_all.append(x); ny_all.append(y); nz_all.append(z)
        txt_all.append(f"D{n}")

    traces.append(
        go.Scatter3d(
            x=nx_all, y=ny_all, z=nz_all,
            mode="markers",
            marker=dict(size=4),
            name="detectors",
            opacity=0.9,
        )
    )

    # ---- 3) 注入 error-event（橙）----
    if injected_edge is not None:
        u, v = injected_edge
        if u in pos and v in pos and u != B and v != B:
            x1, y1, z1 = pos[u]
            x2, y2, z2 = pos[v]
            traces.append(
                go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2],
                    mode="lines",
                    line=dict(width=10),
                    name="injected error-event",
                )
            )

    # ---- 4) fired detectors（蓝点+标签）----
    fx, fy, fz, ftext = [], [], [], []
    for d in fired:
        if d in pos:
            x, y, z = pos[d]
            fx.append(x); fy.append(y); fz.append(z)
            ftext.append(f"D{d}")

    traces.append(
        go.Scatter3d(
            x=fx, y=fy, z=fz,
            mode="markers+text",
            text=ftext,
            textposition="top center",
            marker=dict(size=7),
            name="fired detectors",
        )
    )

    # ---- 5) MWPM correction（红/绿）----
    def pack(color: str):
        xs, ys, zs = [], [], []
        for u, v, c in highlight_edges:
            if c != color:
                continue
            if u not in pos or v not in pos:
                continue
            x1, y1, z1 = pos[u]
            x2, y2, z2 = pos[v]
            xs += [x1, x2, None]
            ys += [y1, y2, None]
            zs += [z1, z2, None]
        return xs, ys, zs

    gx, gy, gz = pack("green")
    rx, ry, rz = pack("red")

    if gx:
        traces.append(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(width=8), name="MWPM correction"))
    if rx:
        traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(width=10), name="logical-crossing segments"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        height=750,
    )
    fig.show()

def _get_stim_detector_coords(circuit: stim.Circuit) -> dict[int, tuple[float, float, float]]:
    """
    Return detector index -> (x,y,z). Missing dims are padded with 0.
    """
    # 1) newer stim usually has circuit.get_detector_coordinates()
    if hasattr(circuit, "get_detector_coordinates"):
        raw = circuit.get_detector_coordinates()  # {det_id: [coords...]}
    else:
        # 2) fallback: from dem (some versions expose it)
        dem = circuit.detector_error_model(decompose_errors=True)
        if hasattr(dem, "get_detector_coordinates"):
            raw = dem.get_detector_coordinates()
        else:
            raw = {}

    coords = {}
    for k, v in raw.items():
        arr = list(v)
        while len(arr) < 3:
            arr.append(0.0)
        coords[int(k)] = (float(arr[0]), float(arr[1]), float(arr[2]))
    return coords


def demo_single_error_event_on_matchgraph(circuit: stim.Circuit, *, seed: int = 1) -> Dict[str, Any]:
    """Demo: inject a *single* DEM error-event edge → fired detectors → MWPM correction."""
    dem = circuit.detector_error_model(decompose_errors=True)
    G, edge_table = build_graph_and_edge_table_from_dem_text(dem)

    rng = random.Random(seed)
    e = rng.choice(edge_table)

    u, v, p, obs_mask = e["u"], e["v"], e["p"], e["obs_mask"]

    fired = []
    if u != "B":
        fired.append(u)
    if v != "B":
        fired.append(v)

    pairs, pair_paths = _mwpm_pairs_with_boundary(G, fired)

    # Highlight the chosen correction path; color segments that flip L0 as red
    highlight = []
    for a, b in pairs:
        path = pair_paths[(a, b)]
        for i in range(len(path) - 1):
            x, y = path[i], path[i + 1]
            if x == "B" or y == "B":
                continue
            flips_obs0 = (G[x][y].get("obs_mask", 0) & 1) != 0
            highlight.append((int(x), int(y), "red" if flips_obs0 else "green"))

    stim_pos = _get_stim_detector_coords(circuit)

    # boundary 点如果你也想显示，可以人为放在图外（可选）
    # stim_pos["B"] = (min_x-1, min_y-1, min_z-1)

    _plot_3d_graph_plotly(
        G,
        pos=stim_pos,
        fired=fired,
        highlight_edges=highlight,
        injected_edge=(u, v),
        title="Matchgraph (stim coords): injected error-event → fired detectors → MWPM correction",
    )

    return {
        "injected_edge": dict(u=u, v=v, p=p, obs_mask=obs_mask),
        "fired_detectors": fired,
        "pairs": pairs,
    }


def explain_shot_on_matchgraph(
    circuit: stim.Circuit,
    *,
    shots: int = 500,
    seed: int = 123,
    prefer_nonzero: bool = True,
) -> Dict[str, Any]:
    """Explain a real sampled shot: fired detectors → MWPM correction (no coords)."""
    dem = circuit.detector_error_model(decompose_errors=True)
    G, _ = build_graph_and_edge_table_from_dem_text(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)

    # choose a shot index
    k = 0
    if prefer_nonzero:
        for i in range(shots):
            if dets[i].any():
                k = i
                break

    fired = np.flatnonzero(dets[k]).tolist()

    pairs, pair_paths = _mwpm_pairs_with_boundary(G, fired)

    highlight = []
    for a, b in pairs:
        path = pair_paths[(a, b)]
        for i in range(len(path) - 1):
            x, y = path[i], path[i + 1]
            if x == "B" or y == "B":
                continue
            flips_obs0 = (G[x][y].get("obs_mask", 0) & 1) != 0
            highlight.append((int(x), int(y), "red" if flips_obs0 else "green"))

    stim_pos = _get_stim_detector_coords(circuit)

    # boundary 点如果你也想显示，可以人为放在图外（可选）
    # stim_pos["B"] = (min_x-1, min_y-1, min_z-1)

    _plot_3d_graph_plotly(
        G,
        pos=stim_pos,
        fired=fired,
        highlight_edges=highlight,
        injected_edge= None,
        title="Matchgraph (stim coords): injected error-event → fired detectors → MWPM correction",
    )

    return {
        "shot": k,
        "fired_detectors": fired,
        "observables": obs[k].astype(int).tolist(),
        "pairs": pairs,
    }

def _sample_one_syndrome_fired_detectors(circuit, *, shots=1, shot=0, seed=None):
    import numpy as np

    sampler = circuit.compile_detector_sampler()

    # stim 版本兼容：有的版本 sample 不支持 seed 参数
    try:
        if seed is None:
            dets = sampler.sample(shots=shots)
        else:
            dets = sampler.sample(shots=shots, seed=seed)
    except TypeError:
        # fallback: older stim
        dets = sampler.sample(shots=shots)

    if shot < 0 or shot >= dets.shape[0]:
        raise ValueError(f"shot index out of range: shot={shot}, shots={shots}")

    fired = np.flatnonzero(dets[shot]).tolist()
    return fired, dets[shot]


def demo_syndrome_on_matchgraph(
    circuit,
    *,
    seed=None,
    shot=0,
    shots=1,
    title=None,
    show_all_detectors=True,
    max_fired_labels=40,
):
    import pymatching
    import networkx as nx

    # 1) DEM
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_text = str(dem)

    # 2) Build matchgraph + edge table (reuse your existing builder)
    G, edge_table = build_graph_and_edge_table_from_dem_text(dem_text)

    # 3) stim coords (reuse your existing coord getter; if not exist, add it)
    pos = _get_stim_detector_coords(circuit)  # detector_id -> (x,y,z)
    # optional: handle boundary node if you use "B"
    # pos["B"] = (...)

    # 4) sample one syndrome
    fired, det_vec = _sample_one_syndrome_fired_detectors(
        circuit, shots=shots, shot=shot, seed=seed
    )

    # 5) MWPM decode (pairs)
    m = pymatching.Matching.from_detector_error_model(dem)
    if hasattr(m, "decode_to_edges_array"):
        pairs = m.decode_to_edges_array(det_vec)
    else:
        # fallback: decode gives correction bit flips, but edges array is nicer for visualization
        # If your pymatching is too old, you can skip edge-highlights or upgrade pymatching.
        pairs = []

    # 6) convert pairs -> highlight edges along graph shortest paths (reuse your existing logic)
    def _canon_node(n):
        # pymatching 用 -1 表示 boundary；你图里用 "B"
        return "B" if n == -1 else n
    highlight_edges = []
    for u, v in pairs:
        u = _canon_node(u)
        v = _canon_node(v)

        # 如果图里没有 boundary 节点（或某节点不存在），就跳过，别让 demo 炸
        if u not in G or v not in G:
            continue

        # 有些图可能不连通，避免 shortest_path 再炸一次
        if not nx.has_path(G, u, v):
            continue

    highlight_edges = []
    skip_missing = 0
    skip_nopath = 0
    use = 0
    for u, v in pairs:
        u = _canon_node(u)
        v = _canon_node(v)

        if u not in G or v not in G:
            continue
        if not nx.has_path(G, u, v):
            continue

        path = nx.shortest_path(G, u, v, weight="weight")

#-----------DEBUG INFO----------------

        if u not in G or v not in G:
            skip_missing += 1
            continue
        if not nx.has_path(G, u, v):
            skip_nopath += 1
            continue
        use += 1
#-----------DEBUG INFO----------------

        # 画整条 MWPM correction（绿），并把 logical-crossing 段标红（覆盖）
        for a, b in zip(path, path[1:]):
            # boundary 段目前不画
            if a == "B" or b == "B":
                continue

            # 1) 默认：普通纠错边（绿）
            highlight_edges.append((int(a), int(b), "green"))

            # 2) 如果这条边会翻转 logical observable L0：再画一遍红色（logical-crossing）
            flips_obs0 = (G[a][b].get("obs_mask", 0) & 1) != 0
            if flips_obs0:
                highlight_edges.append((int(a), int(b), "red"))

    if title is None:
        title = f"Matchgraph (syndrome-level): shot={shot}, fired={len(fired)}"

    # Plot full graph + fired + MWPM correction (no injected edge)
    _plot_3d_graph_plotly(
        G,
        pos=pos,
        fired=fired[:max_fired_labels],   # 防止标签爆炸
        highlight_edges=highlight_edges,
        injected_edge=None,
        title=title,
    )
    #---------------DEBUG INFO----------------
    print("num fired =", len(fired), "num mwpm pairs =", len(pairs), "num highlight edges =", len(highlight_edges))
    n_green = sum(1 for _,_,c in highlight_edges if c=="green")
    n_red   = sum(1 for _,_,c in highlight_edges if c=="red")
    print("edges:", "green=", n_green, "red(logical)=", n_red)


    print("pairs used:", use, "skip_missing:", skip_missing, "skip_nopath:", skip_nopath)


    #---------------RETURN DATA----------------
    return {
        "fired": fired,
        "pairs": pairs,
    }
