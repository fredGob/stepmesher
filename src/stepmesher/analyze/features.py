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


def detect_sliver_seams(face_curves: dict, ref_faces, max_width: float, long_factor: float = 5.0,
                        turn_deg: float = 120.0) -> tuple[list[dict], list[int]]:
    """Faces-lanières de la peau de référence, trop étroites pour porter une rangée d'éléments.

    Certaines CAO CATIA (cadres upper part_002 / 004) portent, au droit des marches
    d'épaisseur, des faces de 0,1 mm de large sur 7 à 25 mm de long en travers des semelles et
    des plis : maillées, elles donnent des arêtes de 0,05 mm (parité) et des hexa écrasés ;
    écrasées par le nettoyage OCP, des faces d'aire nulle que gmsh ne maille pas.
    Lanière = face à une boucle, de largeur moyenne 2 x aire / périmètre < max_width, dont le
    contour se réduit à DEUX chaînes de courbes longues (>= long_factor x max_width) séparées par
    des bouts courts ou un demi-tour (pointe d'une lanière triangulaire, face écrasée à deux
    courbes), de longueurs voisines et aux bouts rapprochés. Seules comptent celles dont les deux
    chaînes bordent la peau de référence : la lanière est alors retirée de la peau et ses deux
    chaînes sont « cousues » au maillage (mêmes nombres de nœuds, nœuds soudés deux à deux,
    mesh.quad.weld_seams). Des lanières accolées font une seule couture entre les chaînes extrêmes.

    Requêtes gmsh sur le modèle courant (toutes les faces). Renvoie (coutures, faces retirées) ;
    couture = dict(face, faces, a=[courbes], b=[courbes], width, length)."""
    import gmsh

    from ..mesh.quad import _ordered_loop, _traversal_tangents

    if max_width <= 0:
        return [], []
    ref = set(int(f) for f in ref_faces)
    curve_faces: dict[int, set] = {}
    for f, cs in face_curves.items():
        for c in cs:
            curve_faces.setdefault(int(c), set()).add(int(f))
    length: dict[int, float] = {}

    def clen(c):
        if c not in length:
            length[c] = gmsh.model.occ.getMass(1, c)
        return length[c]

    def pt(p):
        return np.array(gmsh.model.getValue(0, p, []), float)

    cands: dict[int, dict] = {}
    for f, cs in face_curves.items():
        f = int(f)
        if len(cs) < 2:
            continue
        try:
            per = sum(clen(int(c)) for c in cs)
            try:
                area = abs(gmsh.model.occ.getMass(2, f))
            except Exception:  # noqa: BLE001 - face écrasée
                area = 0.0
            w = 2.0 * area / max(per, 1e-30)
            if w >= max_width or len(gmsh.model.occ.getCurveLoops(f)[0]) != 1:
                continue
            loop, ends = _ordered_loop(f)
            if loop is None:
                continue
            n = len(loop)
            is_long = [clen(c) >= long_factor * max_width for c in loop]
            if sum(is_long) < 2:
                continue
            tans = [_traversal_tangents(c, a) for c, (a, _) in zip(loop, ends)]
            # coupure après la courbe i : courbe courte, ou demi-tour entre deux courbes longues
            cut = []
            for i in range(n):
                j = (i + 1) % n
                turn = float(np.degrees(np.arccos(np.clip(tans[i][1] @ tans[j][0], -1.0, 1.0))))
                cut.append(not is_long[i] or not is_long[j] or turn > turn_deg)
            if not any(cut):
                continue
            start = (next(i for i in range(n) if cut[i]) + 1) % n
            runs, cur = [], []
            for d in range(n):
                i = (start + d) % n
                if is_long[i]:
                    cur.append(i)
                if cut[i] and cur:
                    runs.append(cur)
                    cur = []
            if len(runs) != 2:
                continue
            la, lb = (sum(clen(loop[i]) for i in r) for r in runs)
            if abs(la - lb) > 2.0 * max_width + 0.02 * max(la, lb):
                continue
            # bouts : la fin d'une chaîne fait face au début de l'autre
            a0, a1 = pt(ends[runs[0][0]][0]), pt(ends[runs[0][-1]][1])
            b0, b1 = pt(ends[runs[1][0]][0]), pt(ends[runs[1][-1]][1])
            if max(np.linalg.norm(a1 - b0), np.linalg.norm(b1 - a0)) > 3.0 * max_width:
                continue
            cands[f] = dict(face=f, a=[int(loop[i]) for i in runs[0]], b=[int(loop[i]) for i in runs[1]],
                            width=float(w), length=float(0.5 * (la + lb)))
        except Exception:  # noqa: BLE001
            continue

    def beyond(f, chain):
        """Faces de l'autre côté d'une chaîne : ('ref', None) | ('sliver', g) | (None, None)."""
        others = set()
        for c in chain:
            o = curve_faces.get(c, set()) - {f}
            if len(o) != 1:
                return None, None
            others |= o
        if others <= (ref - set(cands)):
            return "ref", None
        if len(others) == 1:
            g = next(iter(others))
            if g in cands and (set(cands[g]["a"]) == set(chain) or set(cands[g]["b"]) == set(chain)):
                return "sliver", g
        return None, None

    seams, removed, seen = [], [], set()
    for f in sorted(cands):
        if f in seen:
            continue
        # lanières accolées : on avance de chaque côté jusqu'à une face de peau ordinaire
        group, outer, ok = [f], [], True
        for key in ("a", "b"):
            g, chain = f, cands[f][key]
            for _ in range(8):
                kind, nxt = beyond(g, chain)
                if kind == "ref":
                    outer.append(list(chain))
                    break
                if kind != "sliver" or nxt in group:
                    ok = False
                    break
                group.append(nxt)
                chain = cands[nxt]["b"] if set(cands[nxt]["a"]) == set(chain) else cands[nxt]["a"]
                g = nxt
            else:
                ok = False
            if not ok:
                break
        if not ok or len(outer) != 2:
            continue
        seen.update(group)
        removed += group
        seams.append(dict(face=int(f), faces=sorted(int(g) for g in group), a=outer[0], b=outer[1],
                          width=float(sum(cands[g]["width"] for g in group)),
                          length=float(cands[f]["length"])))
    return seams, sorted(set(removed))
