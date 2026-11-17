"""Post-run trajectory quality control for LiMOSAT products.

LiMOSAT QC (protocol v1: one sequential pass over the matching output) is in ``limosat.qc.qc`` and runs
with ``python -m limosat.qc``. The earlier v2 streaming QC (``limosat.qc.archive``) and the v3 drafts
(``limosat.qc.v3``) are kept for reproducing earlier releases; ``limosat.qc.qc`` reuses their neighbour
model, land raster and sea-ice samplers.
"""

from .core import (
    PROTOCOL_ID,
    PROTOCOL_PATH,
    QCConfig,
    load_protocol,
    score_edges,
    score_vectors,
    validate_frozen_protocol,
)

__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_PATH",
    "QCConfig",
    "load_protocol",
    "score_edges",
    "score_vectors",
    "validate_frozen_protocol",
]
