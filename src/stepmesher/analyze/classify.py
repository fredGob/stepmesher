"""Classification de la pièce et choix de la peau de référence."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..occ.topology import SurfTri
from .skins import SkinAnalysis, _components

CONSTANT, VARIABLE, MASSIVE = "constant", "variable", "massive"


def thickness_zones(values: list[float], rel_tol: float) -> list[float]:
    """Regroupe des épaisseurs en paliers (écart relatif > rel_tol entre paliers)."""
    v = sorted(x for x in values if np.isfinite(x) and x > 0)
    if not v:
        return []
    zones = [[v[0]]]
    for x in v[1:]:
        if x - zones[-1][-1] > rel_tol * zones[-1][-1]:
            zones.append([x])
        else:
            zones[-1].append(x)
    return [float(np.median(z)) for z in zones]


def thickness_zones_weighted(t: np.ndarray, w: np.ndarray, rel_tol: float, min_area_frac: float = 0.02):
    """Paliers d'épaisseur pondérés par l'aire ; les paliers < min_area_frac sont ignorés."""
    m = np.isfinite(t) & (t > 0)
    t, w = t[m], w[m]
    if len(t) == 0:
        return []
    # histogramme pondéré en log(t) : un palier = suite de classes « peuplées » ;
    # les rampes de transition (peu d'aire par classe) séparent les paliers.
    lt = np.log(t)
    bw = np.log1p(rel_tol) / 2
    edges = np.arange(lt.min() - bw, lt.max() + 2 * bw, bw)
    mass, _ = np.histogram(lt, bins=edges, weights=w)
    occ = mass >= 0.005 * w.sum()
    zones = []
    i = 0
    while i < len(occ):
        if not occ[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(occ) and occ[j + 1]:
            j += 1
        sel = (lt >= edges[i]) & (lt < edges[j + 1])
        if w[sel].sum() >= min_area_frac * w.sum():
            zones.append(float(np.exp(np.average(lt[sel], weights=w[sel]))))
        i = j + 1
    return zones


def classify(sk: SkinAnalysis, cfg) -> dict:
    a = cfg["analysis"]
    tol = a["constant_thickness_rel_tol"]
    if sk.skin_area_fraction < 0.5 or not np.isfinite(sk.t_median) or sk.t_median <= 0:
        kind = MASSIVE
        disp = float("nan")
    else:
        disp = (sk.t_p90 - sk.t_p10) / sk.t_median
        kind = CONSTANT if disp <= tol else VARIABLE
    return dict(kind=kind, dispersion=disp, t_median=sk.t_median, t_p10=sk.t_p10, t_p90=sk.t_p90,
                skin_area_fraction=sk.skin_area_fraction)


def side_components(st: SurfTri, sk: SkinAnalysis, side: int) -> int:
    nb = st.neighbors()
    faces = [f for f, s in sk.side_of_face.items() if s == side]
    fs = set(faces)
    _, comps = _components(sorted(faces), {f: [g for g in nb.get(f, ()) if g in fs] for f in faces})
    return len(comps)


def choose_reference_side(st: SurfTri, sk: SkinAnalysis, kind: str, mode: str = "inner") -> tuple[int, str]:
    """Peau de référence.

    - tôle constante : peau intérieure (aire la plus faible) pour que le décalage
      diverge aux plis ; `outer` inverse le choix ;
    - épaisseur variable : peau la moins découpée (le moins de faces CAD), c'est-à-dire
      la face non usinée / non fraisée : son maillage quad est propre et les marches
      d'épaisseur sont reportées sur l'autre peau.
    """
    if kind == VARIABLE:
        n0 = sum(1 for s in sk.side_of_face.values() if s == 0)
        n1 = sum(1 for s in sk.side_of_face.values() if s == 1)
        if min(n0, n1) < 0.8 * max(n0, n1):
            return (0 if n0 < n1 else 1), f"peau la moins découpée ({min(n0, n1)} faces contre {max(n0, n1)})"
        c0, c1 = side_components(st, sk, 0), side_components(st, sk, 1)
        if c0 != c1:
            return (0 if c0 < c1 else 1), f"peau la moins fragmentée ({c0} vs {c1} composantes)"
    inner = 0 if sk.side_area[0] <= sk.side_area[1] else 1
    if mode == "outer":
        return 1 - inner, "peau extérieure (option)"
    return inner, "peau intérieure (aire la plus faible)"
