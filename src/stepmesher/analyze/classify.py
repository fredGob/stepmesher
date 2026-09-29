"""Classification de la pièce et choix de la peau de référence."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..occ.topology import SurfTri
from .skins import SkinAnalysis, _components

CONSTANT, VARIABLE, MASSIVE, PROFILE = "constant", "variable", "massive", "profile"


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


def classify(sk: SkinAnalysis, cfg, st: SurfTri | None = None, obb_center=None,
            obb_axes=None, obb_dims=None) -> dict:
    a = cfg["analysis"]
    tol = a["constant_thickness_rel_tol"]
    if sk.skin_area_fraction < 0.5 or not np.isfinite(sk.t_median) or sk.t_median <= 0:
        kind = MASSIVE
        disp = float("nan")
    else:
        disp = (sk.t_p90 - sk.t_p10) / sk.t_median
        kind = CONSTANT if disp <= tol else VARIABLE
    out = dict(kind=kind, dispersion=disp, t_median=sk.t_median, t_p10=sk.t_p10, t_p90=sk.t_p90,
               skin_area_fraction=sk.skin_area_fraction)
    if kind != MASSIVE and st is not None and obb_axes is not None and obb_dims is not None:
        prof = detect_swept_profile(st, obb_center, obb_axes, obb_dims, cfg)
        out["profile"] = prof
        if prof["is_profile"]:
            out["thickness_kind"] = kind
            out["kind"] = kind = PROFILE
    return out


def _section_perimeter(P: np.ndarray, T: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    """Longueur totale des segments d'intersection triangle/plan (périmètre de la section)."""
    d = (P - origin) @ axis
    edges = ((0, 1), (1, 2), (2, 0))
    n = len(T)
    cross = np.zeros((3, n), bool)
    pts = np.zeros((3, n, 3))
    for k, (i, j) in enumerate(edges):
        di, dj = d[T[:, i]], d[T[:, j]]
        c = (di * dj) < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(c, di / (di - dj), 0.0)
        pi, pj = P[T[:, i]], P[T[:, j]]
        pts[k] = pi + t[:, None] * (pj - pi)
        cross[k] = c
    valid = cross.sum(axis=0) == 2
    if not valid.any():
        return 0.0
    idx = np.arange(n)
    first = np.argmax(cross, axis=0)
    cross2 = cross.copy()
    cross2[first, idx] = False
    second = np.argmax(cross2, axis=0)
    seg = np.linalg.norm(pts[first, idx] - pts[second, idx], axis=1)
    return float(seg[valid].sum())


def detect_swept_profile(st: SurfTri, obb_center, obb_axes, obb_dims, cfg) -> dict:
    """Détecte un profilé/lisse (T, Z, L, oméga...) : section quasi constante balayée le long
    de son axe long, par opposition à une tôle pliée ponctuellement.

    Coupe la triangulation d'analyse par plusieurs plans perpendiculaires à l'axe long (le
    premier axe OBB) et compare le périmètre de section obtenu le long des stations (dispersion
    relative p90/p10 vs médiane, comme pour l'épaisseur). Détection seule pour l'instant : pas
    encore de stratégie de maillage par balayage dédiée à ce type.
    """
    a = cfg["analysis"]
    signals = dict(elongation=0.0, section_dispersion=float("nan"), n_stations=0)
    if obb_dims[1] <= 1e-9:
        return dict(is_profile=False, **signals)
    elongation = float(obb_dims[0] / obb_dims[1])
    signals["elongation"] = round(elongation, 3)
    if elongation < a["profile_elongation_min"]:
        return dict(is_profile=False, **signals)
    axis = np.asarray(obb_axes[0], float)
    center = np.asarray(obb_center, float)
    n_stations = int(a["profile_n_stations"])
    margin = float(a["profile_margin_frac"])
    us = np.linspace(-0.5 + margin, 0.5 - margin, n_stations)
    perims = np.array([_section_perimeter(st.P, st.T, center + u * obb_dims[0] * axis, axis) for u in us])
    perims = perims[np.isfinite(perims) & (perims > 0)]
    signals["n_stations"] = int(len(perims))
    if len(perims) < max(3, n_stations // 2):
        return dict(is_profile=False, **signals)
    p10, p50, p90 = np.percentile(perims, [10, 50, 90])
    disp = float((p90 - p10) / p50) if p50 > 0 else float("inf")
    signals["section_dispersion"] = round(disp, 4)
    return dict(is_profile=disp <= a["profile_section_rel_tol"], **signals)


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
