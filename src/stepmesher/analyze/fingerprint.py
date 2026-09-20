"""Empreinte d'une pièce : hash exact invariant + vecteur de caractéristiques.

Le hash ne dépend que de grandeurs invariantes par déplacement rigide (nombres et
types de faces, volume, aire, moments principaux d'inertie au centre de masse),
arrondies à 4 chiffres significatifs.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import gmsh
import numpy as np


def _sig(x: float, n: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def exact_invariants() -> dict:
    vols = [t for _, t in gmsh.model.getEntities(3)]
    faces = [t for _, t in gmsh.model.getEntities(2)]
    types = Counter(gmsh.model.getType(2, t) for t in faces)
    V = sum(gmsh.model.occ.getMass(3, t) for t in vols)
    A = sum(gmsh.model.occ.getMass(2, t) for t in faces)
    I = np.zeros((3, 3))
    for t in vols:
        I += np.array(gmsh.model.occ.getMatrixOfInertia(3, t)).reshape(3, 3)
    # plusieurs volumes : matrices prises chacune en leur centre de masse -> reste invariant
    moments = sorted(np.linalg.eigvalsh(0.5 * (I + I.T)).tolist())
    return dict(n_solids=len(vols), n_faces=len(faces), face_types=dict(sorted(types.items())),
                volume=_sig(V), area=_sig(A), moments=[_sig(m) for m in moments])


def exact_hash(inv: dict) -> str:
    return hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest()[:32]


FEATURE_NAMES = ["is_sheet", "log_t_median", "t_spread", "log_diag", "aspect_21", "aspect_31",
                 "n_bends", "n_holes", "min_bend_r_over_t", "min_hole_d_over_t", "n_zones"]


def feature_vector(kind: str, cls: dict, diag: float, dims, n_bends: int, n_holes: int,
                   min_bend_r: float, min_hole_d: float, n_zones: int) -> list[float]:
    t = cls["t_median"] if np.isfinite(cls["t_median"]) and cls["t_median"] > 0 else 1.0
    d = sorted(dims, reverse=True)
    return [
        1.0 if kind in ("constant", "variable") else 0.0,
        math.log(t),
        float(cls["dispersion"]) if np.isfinite(cls["dispersion"]) else 0.0,
        math.log(max(diag, 1e-9)),
        d[1] / max(d[0], 1e-9),
        d[2] / max(d[0], 1e-9),
        math.log1p(n_bends),
        math.log1p(n_holes),
        math.log1p(min_bend_r / t) if np.isfinite(min_bend_r) else 0.0,
        math.log1p(min_hole_d / t) if np.isfinite(min_hole_d) else 0.0,
        float(n_zones),
    ]
