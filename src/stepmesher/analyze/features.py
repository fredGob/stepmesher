"""Features géométriques : trous, plis, arêtes vives, bords libres.

Tout est calculé sur la triangulation d'analyse orientée (pas de requête OCC).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..occ.topology import SurfTri
from .skins import SkinAnalysis, _components


def _tri_index(st: SurfTri) -> dict[int, np.ndarray]:
    d = defaultdict(list)
    for i, f in enumerate(st.face.tolist()):
        d[f].append(i)
    return {f: np.array(v) for f, v in d.items()}


def detect_holes(st: SurfTri, sk: SkinAnalysis, t_default: float) -> list[dict]:
    """Trous débouchants : composantes connexes de chants dont les normales sortantes
    convergent vers leur centre (paroi concave), à la différence du contour extérieur."""
    nb = st.neighbors()
    flank_nb = {f: [g for g in nb.get(f, ()) if g in sk.flank_faces] for f in sk.flank_faces}
    _, comps = _components(sorted(sk.flank_faces), flank_nb)
    tri_of = _tri_index(st)
    holes = []
    for comp in comps:
        idx = np.concatenate([tri_of[f] for f in comp if f in tri_of]) if comp else np.zeros(0, int)
        if len(idx) == 0:
            continue
        w = st.area[idx]
        A = float(w.sum())
        c = np.average(st.centroid[idx], axis=0, weights=w)
        r = c - st.centroid[idx]
        rn = np.linalg.norm(r, axis=1)
        conc = float(np.average(np.einsum("ij,ij->i", st.normal[idx], r) / np.maximum(rn, 1e-12), weights=w))
        # épaisseur locale : faces de peau voisines
        ts = [sk.face_thickness[g] for f in comp for g in nb.get(f, ()) if g in sk.face_thickness
              and np.isfinite(sk.face_thickness[g])]
        t_loc = float(np.median(ts)) if ts else t_default
        touches = {sk.side_of_face[g] for f in comp for g in nb.get(f, ()) if g in sk.side_of_face}
        N = st.normal[idx] * np.sqrt(w)[:, None]
        _, _, Vt = np.linalg.svd(N, full_matrices=False)
        axis = Vt[-1]
        # diamètre : étendue des points dans le plan perpendiculaire à l'axe
        X = st.centroid[idx] - c
        Xp = X - np.outer(X @ axis, axis)
        diam_ext = 2 * float(np.average(np.linalg.norm(Xp, axis=1), weights=w))
        diam_eq = A / max(t_loc, 1e-12) / np.pi
        is_hole = conc > 0.3 and len(touches) == 2
        holes.append(dict(faces=sorted(comp), area=A, center=c.tolist(), axis=axis.tolist(),
                          diameter=float(min(diam_eq, diam_ext) if is_hole else diam_eq),
                          concavity=conc, thickness=t_loc, is_hole=bool(is_hole)))
    return holes


def detect_bends(st: SurfTri, sk: SkinAnalysis, side: int, cfg) -> tuple[list[dict], list[dict]]:
    """Plis (faces courbes à faible rayon) et arêtes vives sur la peau `side`."""
    a = cfg["analysis"]
    tri_of = _tri_index(st)
    faces = [f for f, s in sk.side_of_face.items() if s == side]
    curved = {}
    for f in faces:
        idx = tri_of.get(f)
        if idx is None or len(idx) < 3:
            continue
        w = st.area[idx]
        N = st.normal[idx]
        m = np.average(N, axis=0, weights=w)
        m /= max(np.linalg.norm(m), 1e-12)
        spread = 2 * float(np.degrees(np.arccos(np.clip(N @ m, -1, 1))).max())
        if spread < a["bend_min_angle_deg"]:
            continue
        _, _, Vt = np.linalg.svd(N * np.sqrt(w)[:, None], full_matrices=False)
        axis = Vt[-1]
        proj = st.centroid[idx] @ axis
        L = float(proj.max() - proj.min())
        area = float(w.sum())
        theta = np.radians(spread)
        R = area / max(theta * L, 1e-12) if L > 0 else float("inf")
        t = sk.face_thickness.get(f, np.nan)
        # rayon de pli de tôlerie : quelques épaisseurs, pas une grande courbure de peau
        if np.isfinite(t) and R <= 25 * t:
            curved[f] = dict(face=f, angle_deg=spread, radius=R, axis=axis, length=L, area=area)

    # regroupement des faces courbes adjacentes et coaxiales en un seul pli
    nb = st.neighbors()
    cnb = {f: [g for g in nb.get(f, ()) if g in curved and abs(curved[f]["axis"] @ curved[g]["axis"]) > 0.95]
           for f in curved}
    _, comps = _components(sorted(curved), cnb)
    bends = []
    for comp in comps:
        ang = sum(curved[f]["angle_deg"] for f in comp) / max(1, len({round(curved[f]["length"], 0) for f in comp}))
        bends.append(dict(faces=sorted(comp), angle_deg=float(min(ang, 180.0)),
                          radius=float(np.median([curved[f]["radius"] for f in comp])),
                          length=float(max(curved[f]["length"] for f in comp)),
                          axis=curved[comp[0]]["axis"].tolist()))

    sharp = []
    fs = set(faces)
    for (x, y), v in st.adjacency.items():
        if x in fs and y in fs and v["angle_deg"] > a["sharp_edge_angle_deg"]:
            sharp.append(dict(faces=[x, y], angle_deg=v["angle_deg"], length=v["length"],
                              convex=v["convex"] > 0.5))
    return bends, sharp


def free_edge_length(st: SurfTri, sk: SkinAnalysis, side: int) -> float:
    """Longueur des bords de la peau `side` (arêtes peau/chant)."""
    ta, tb = st.edge_tris[:, 0], st.edge_tris[:, 1]
    fa, fb = st.face[ta], st.face[tb]
    side_faces = np.array([f for f, s in sk.side_of_face.items() if s == side])
    flank = np.array(sorted(sk.flank_faces)) if sk.flank_faces else np.zeros(0, int)
    m = (np.isin(fa, side_faces) & np.isin(fb, flank)) | (np.isin(fb, side_faces) & np.isin(fa, flank))
    return float(st.edge_len[m].sum())
