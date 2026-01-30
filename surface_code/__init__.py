"""
Surface-code utilities (Stim-based).

Organized for:
- Circuit generation (Stim)
- DEM utilities (Stim diagrams + sampling)
- Decoding (PyMatching)
- Match-graph explanation without relying on coords (NetworkX + Plotly)
"""

from .stim_circuit import (
    SurfaceCodeParams,
    build_surface_code_circuit,
    circuit_text,
    circuit_ascii_diagram,
    draw_chip_topology, 
    chip_topology_from_circuit, 
    qubit_coords_from_circuit
)

from .stim_dem import (
    detector_error_model,
    detector_matchgraph,
    sample_detectors_and_observables,
)

from .decoding import (
    estimate_logical_error_rate_mwpm,
)

from .dem_graph import (
    build_graph_and_edge_table_from_dem_text,
)

from .annotate_no_coords import (
    demo_single_error_event_on_matchgraph,
    explain_shot_on_matchgraph,
    demo_syndrome_on_matchgraph, 
)
