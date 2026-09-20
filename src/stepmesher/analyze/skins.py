"""Détection des peaux et de l'épaisseur locale.

Méthode (aucune requête OCC point par point) :
1. depuis le centre de chaque triangle de la triangulation d'analyse, on lance un
   rayon vers l'intérieur de la matière (-normale sortante) ;
2. les triangles candidats sont trouvés par KDTree, par classes de taille (sinon
   les grands triangles des faces planes imposeraient un rayon de recherche énorme
   à toute la pièce), puis intersection exacte Möller-Trumbore vectorisée ;
3. premier impact = face opposée et distance = épaisseur locale ;
4. une face est une peau si son épaisseur médiane est petite devant sa largeur
   propre 2*aire/périmètre (sinon : chant, paroi de trou) ; la décision se propage
   aux faces tangentes d'épaisseur cohérente (plis, lanières CATIA étroites) mais
   pas à travers les congés d'arête peau/chant (couverture antiparallèle faible) ;
5. les faces de peau sont réparties en deux côtés par 2-coloration de la
   relation « face opposée ».
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..occ.topology import SurfTri

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# lancer de rayons
# ----------------------------------------------------------------------------
def _moller_trumbore(o, d, v0, v1, v2, eps_t):
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14 * np.maximum(np.einsum("ij,ij->i", e1, e1), 1e-30)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - v0
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    tol = 1e-7
    ok &= (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol) & (t > eps_t)
    return ok, t


def cast_rays(st: SurfTri, origins: np.ndarray, dirs: np.ndarray, t_max: float,
              exclude: np.ndarray | None = None, chunk_rays: int = 20000, exit_cos: float | None = None):
    """Premier impact de chaque rayon (origins, dirs) sur la triangulation, à distance <= t_max.

    Recherche des candidats dans un *tube* autour du segment de rayon (points
    échantillonnés le long du rayon), par classes de taille de triangles, et par
    blocs de rayons : mémoire bornée même sur les pièces très percées.
    `exit_cos` : si donné, seuls comptent les triangles par lesquels le rayon sort
    de la matière (normale sortante · direction > exit_cos) ; écarte les impacts
    rasants et les parois rentrantes.
    Retourne (t (n,), tri (n,)) ; t = inf et tri = -1 sans impact.
    """
    n = len(origins)
    best_t = np.full(n, np.inf)
    best_j = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(st.T) == 0:
        return best_t, best_j
    P, T = st.P, st.T
    sizes = st.size
    eps_t = 1e-6 * max(t_max, 1e-9)
    smin = max(float(sizes.min()), 1e-9)
    kmin = int(np.floor(np.log2(smin / t_max)))
    kmax = int(np.ceil(np.log2(max(float(sizes.max()), smin) / t_max)))
    classes = []
    for k in range(kmin, kmax + 1):
        lo, hi = t_max * 2.0 ** (k - 1), t_max * 2.0 ** k
        J = np.nonzero((sizes > lo) & (sizes <= hi))[0] if k > kmin else np.nonzero(sizes <= hi)[0]
        if len(J):
            step = max(hi, t_max / 8.0)
            K = int(np.ceil(t_max / step)) + 1
            classes.append((J, cKDTree(st.centroid[J]), hi + 0.5 * step, step, K))

    for s0 in range(0, n, chunk_rays):
        sl = slice(s0, min(n, s0 + chunk_rays))
        o, d = origins[sl], dirs[sl]
        m = len(o)
        for J, tree_j, r, step, K in classes:
            samples = (o[:, None, :] + d[:, None, :] * (np.arange(K) * step)[None, :, None]).reshape(-1, 3)
            sdm = cKDTree(samples).sparse_distance_matrix(tree_j, r, output_type="ndarray")
            if len(sdm) == 0:
                continue
            I = sdm["i"].astype(np.int64) // K
            Jl = sdm["j"].astype(np.int64)
            key = np.unique(I * len(J) + Jl)
            I, Jg = key // len(J), J[key % len(J)]
            Ig = I + s0
            if exclude is not None:
                keep = exclude[Ig] != Jg
                I, Ig, Jg = I[keep], Ig[keep], Jg[keep]
            tri = T[Jg]
            ok, t = _moller_trumbore(o[I], d[I], P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]], eps_t)
            ok &= t <= t_max
            if exit_cos is not None:
                ok &= np.einsum("ij,ij->i", st.normal[Jg], d[I]) > exit_cos
            if not ok.any():
                continue
            Ig, Jg, t = Ig[ok], Jg[ok], t[ok]
            order = np.lexsort((t, Ig))
            Ig, Jg, t = Ig[order], Jg[order], t[order]
            first = np.r_[True, Ig[1:] != Ig[:-1]]
            Ig, Jg, t = Ig[first], Jg[first], t[first]
            better = t < best_t[Ig]
            best_t[Ig[better]] = t[better]
            best_j[Ig[better]] = Jg[better]
    return best_t, best_j


# ----------------------------------------------------------------------------
# résultats
# ----------------------------------------------------------------------------
@dataclass
class SkinAnalysis:
    tri_t: np.ndarray                 # épaisseur mesurée par triangle (inf si aucune)
    tri_opp: np.ndarray               # triangle opposé (-1)
    tri_anti: np.ndarray              # impact sur une face antiparallèle
    patch_of_face: dict[int, int]
    patches: list[dict]
    skin_faces: set[int]
    flank_faces: set[int]
    face_thickness: dict[int, float]
    face_opposite: dict[int, int]
    side_of_face: dict[int, int]      # 0 / 1 pour les faces de peau
    side_area: list[float] = field(default_factory=lambda: [0.0, 0.0])
    side_conflicts: int = 0
    t_median: float = 0.0
    t_p10: float = 0.0
    t_p90: float = 0.0
    skin_area_fraction: float = 0.0


def _weighted_quantile(x, w, q):
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    c = c / c[-1]
    return float(np.interp(q, c, x))


def _components(nodes, edges_ok):
    """Composantes connexes ; edges_ok : dict node -> iterable de voisins."""
    comp = {}
    out = []
    for n0 in nodes:
        if n0 in comp:
            continue
        cid = len(out)
        comp[n0] = cid
        members = [n0]
        q = deque([n0])
        while q:
            a = q.popleft()
            for b in edges_ok.get(a, ()):
                if b not in comp:
                    comp[b] = cid
                    members.append(b)
                    q.append(b)
        out.append(members)
    return comp, out


def analyze_skins(st: SurfTri, cfg, t_max: float) -> SkinAnalysis:
    a = cfg["analysis"]
    cos_anti = np.cos(np.radians(a["opposite_normal_tol_deg"]))
    M = len(st.T)
    t, j = cast_rays(st, st.centroid, -st.normal, t_max, exclude=np.arange(M))
    hit = j >= 0
    anti = np.zeros(M, bool)
    anti[hit] = np.einsum("ij,ij->i", st.normal[hit], st.normal[j[hit]]) < -cos_anti

    # --- statistiques par face ---
    faces = sorted(set(st.face.tolist()))
    fidx = {f: i for i, f in enumerate(faces)}
    tri_f = np.array([fidx[int(f)] for f in st.face])
    nf = len(faces)
    f_area = np.bincount(tri_f, weights=st.area, minlength=nf)
    ta, tb = st.edge_tris[:, 0], st.edge_tris[:, 1]
    border = tri_f[ta] != tri_f[tb]
    f_perim = (np.bincount(tri_f[ta[border]], weights=st.edge_len[border], minlength=nf)
               + np.bincount(tri_f[tb[border]], weights=st.edge_len[border], minlength=nf))

    valid = hit & anti
    tri_idx_of_face = defaultdict(list)
    for i, f in enumerate(st.face.tolist()):
        tri_idx_of_face[f].append(i)

    info = {}
    for f in faces:
        k = fidx[f]
        idx = np.array(tri_idx_of_face[f])
        mv = valid[idx]
        cov = float(st.area[idx[mv]].sum() / max(f_area[k], 1e-30))
        if mv.any():
            d = _weighted_quantile(t[idx[mv]], st.area[idx[mv]], 0.5)
            u, c = np.unique(st.face[j[idx[mv]]], return_counts=True)
            opp = int(u[np.argmax(c)])
        else:
            d, opp = float("inf"), -1
        width = 2 * f_area[k] / f_perim[k] if f_perim[k] > 0 else float("inf")
        info[f] = dict(area=float(f_area[k]), perimeter=float(f_perim[k]), width=float(width),
                       thickness=float(d), coverage=cov, opposite=opp)

    # 1) critère local : épaisseur petite devant la largeur propre de la face
    skin = {f for f, v in info.items() if v["coverage"] >= 0.5 and v["thickness"] < a["skin_ratio"] * v["width"]}

    # 2) propagation : faces tangentes à une peau, d'épaisseur cohérente (plis,
    #    lanières CATIA étroites). Les congés d'arête peau/chant ont une
    #    couverture antiparallèle faible et ne propagent pas.
    def consistent(f, g):
        tf, tg = info[f]["thickness"], info[g]["thickness"]
        return np.isfinite(tf) and np.isfinite(tg) and abs(tf - tg) <= 0.25 * max(tg, 1e-12)

    nb = st.neighbors()
    changed = True
    while changed:
        changed = False
        for f in faces:
            if f in skin or info[f]["coverage"] < 0.5:
                continue
            for g in nb.get(f, ()):
                if g in skin and st.adjacency[(min(f, g), max(f, g))]["angle_deg"] < a["patch_angle_deg"] \
                        and consistent(f, g):
                    skin.add(f)
                    changed = True
                    break
            else:
                o = info[f]["opposite"]
                if o in skin and info[o]["opposite"] == f and consistent(f, o):
                    skin.add(f)
                    changed = True

    skin_faces = skin
    flank_faces = set(faces) - skin_faces
    face_t = {f: info[f]["thickness"] for f in skin_faces}
    face_opp = {f: info[f]["opposite"] for f in skin_faces if info[f]["opposite"] >= 0}
    patch_of_face = {f: fidx[f] for f in faces}
    patches = [dict(id=fidx[f], faces=[f], skin=f in skin_faces, **info[f]) for f in faces]

    # --- côtés : composantes de faces de peau, 2-coloration par la relation "opposé" ---
    nb = st.neighbors()
    skin_nb = {f: [g for g in nb.get(f, ()) if g in skin_faces] for f in skin_faces}
    comp_of, comps = _components(sorted(skin_faces), skin_nb)
    face_area = st.face_area()
    comp_area = [sum(face_area.get(f, 0) for f in c) for c in comps]
    votes = defaultdict(float)
    for f, g in face_opp.items():
        if g in comp_of and comp_of[f] != comp_of[g]:
            x, y = sorted((comp_of[f], comp_of[g]))
            votes[(x, y)] += face_area.get(f, 0)
    cg = defaultdict(list)
    for (x, y), w in votes.items():
        cg[x].append((y, w))
        cg[y].append((x, w))
    color = {}
    conflicts = 0
    for c0 in sorted(range(len(comps)), key=lambda c: -comp_area[c]):
        if c0 in color:
            continue
        color[c0] = 0
        q = deque([c0])
        while q:
            c = q.popleft()
            for d, w in sorted(cg[c], key=lambda x: -x[1]):
                if d not in color:
                    color[d] = 1 - color[c]
                    q.append(d)
                elif color[d] == color[c]:
                    conflicts += 1
    side_of_face = {f: color[comp_of[f]] for f in skin_faces}
    side_area = [0.0, 0.0]
    for f, s in side_of_face.items():
        side_area[s] += face_area.get(f, 0)

    skin_tri = np.isin(st.face, list(skin_faces)) & valid
    res = SkinAnalysis(tri_t=t, tri_opp=j, tri_anti=anti, patch_of_face=patch_of_face,
                       patches=patches, skin_faces=skin_faces, flank_faces=flank_faces,
                       face_thickness=face_t, face_opposite=face_opp, side_of_face=side_of_face,
                       side_area=side_area, side_conflicts=conflicts)
    if skin_tri.any():
        w = st.area[skin_tri]
        res.t_median = _weighted_quantile(t[skin_tri], w, 0.5)
        res.t_p10 = _weighted_quantile(t[skin_tri], w, 0.10)
        res.t_p90 = _weighted_quantile(t[skin_tri], w, 0.90)
    res.skin_area_fraction = float(sum(face_area.get(f, 0) for f in skin_faces) / max(st.area.sum(), 1e-30))
    return res
