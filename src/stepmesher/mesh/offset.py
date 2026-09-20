"""Décalage des nœuds de la peau de référence jusqu'à la peau opposée.

Pour chaque nœud :
1. normale sortante CAD exacte de chaque face incidente (getClosestPoint +
   getNormal, en un appel par face) ; direction de décalage = opposée de la
   moyenne (bissectrice aux arêtes vives) ; facteur d'onglet 1/cos(θ/2) ;
2. rayon lancé sur la triangulation d'analyse de la peau opposée (KDTree +
   Möller-Trumbore) ; à défaut d'impact (bord, chanfrein), point le plus proche ;
3. affinage sur la surface CAD exacte de la face touchée (getClosestPoint).
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from scipy.spatial import cKDTree

from ..analyze.skins import cast_rays
from ..occ.topology import SurfTri
from .quad import QuadMesh


@dataclass
class OffsetResult:
    X_top: np.ndarray        # (N,3)
    direction: np.ndarray    # (N,3) direction de décalage (vers la peau opposée)
    miter: np.ndarray        # (N,)
    t_expected: np.ndarray   # (N,) épaisseur attendue (faces incidentes)
    t_ray: np.ndarray        # (N,) distance de l'impact sur la triangulation
    reproj_err: np.ndarray   # (N,) |point décalé - point reprojeté| / épaisseur attendue
    fallback: np.ndarray     # (N,) bool : pas d'impact direct, point le plus proche
    top_face: np.ndarray     # (N,) face opposée atteinte


def _load_surface(pa, faces) -> SurfTri:
    d = np.load(pa.tri_file)
    m = np.isin(d["face"], faces)
    T = d["T"][m]
    return SurfTri(P=d["P"], T=T, face=d["face"][m], normal=d["normal"][m], area=d["area"][m],
                   centroid=d["centroid"][m], size=d["size"][m], face_tags=list(faces))


def _closest_on_triangles(p, a, b, c):
    """Point le plus proche de p sur les triangles (a,b,c), vectorisé (Ericson)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    v = vb / denom
    w = vc / denom
    res = a + ab * v[:, None] + ac * w[:, None]
    # régions de sommets / arêtes
    r = d1 <= 0
    r &= d2 <= 0
    res[r] = a[r]
    r2 = (d3 >= 0) & (d4 <= d3)
    res[r2] = b[r2]
    r3 = (d6 >= 0) & (d5 <= d6)
    res[r3] = c[r3]
    e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~(r | r2 | r3)
    tt = d1 / np.where(d1 - d3 == 0, 1e-300, d1 - d3)
    res[e_ab] = (a + ab * tt[:, None])[e_ab]
    e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~(r | r2 | r3 | e_ab)
    tt = d2 / np.where(d2 - d6 == 0, 1e-300, d2 - d6)
    res[e_ac] = (a + ac * tt[:, None])[e_ac]
    e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~(r | r2 | r3 | e_ab | e_ac)
    tt = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-300, (d4 - d3) + (d5 - d6))
    res[e_bc] = (b + (c - b) * tt[:, None])[e_bc]
    return res


def nearest_on_surface(st: SurfTri, pts: np.ndarray, k: int = 12):
    tree = cKDTree(st.centroid)
    k = min(k, len(st.T))
    _, J = tree.query(pts, k=k)
    J = J.reshape(len(pts), -1)
    best_d = np.full(len(pts), np.inf)
    best_p = np.zeros_like(pts)
    best_j = np.full(len(pts), -1)
    for col in range(J.shape[1]):
        j = J[:, col]
        tri = st.T[j]
        q = _closest_on_triangles(pts, st.P[tri[:, 0]], st.P[tri[:, 1]], st.P[tri[:, 2]])
        d = np.linalg.norm(q - pts, axis=1)
        b = d < best_d
        best_d[b], best_p[b], best_j[b] = d[b], q[b], j[b]
    return best_p, best_j


def node_normals(pa, qm: QuadMesh):
    """Normales sortantes CAD moyennées par nœud + facteur d'onglet."""
    N = len(qm.X)
    acc = np.zeros((N, 3))
    cnt = np.zeros(N)
    faces_of = {}
    for arr, fa in ((qm.quads, qm.quad_face), (qm.tris, qm.tri_face)):
        for f in np.unique(fa):
            nodes = np.unique(arr[fa == f].ravel())
            faces_of[int(f)] = np.union1d(faces_of.get(int(f), np.zeros(0, np.int64)), nodes)
    per_face_n = []
    projected = np.zeros(N, bool)
    X_new = qm.X.copy()
    t_med = pa.classification["t_median"] if pa.classification["t_median"] > 0 else 1.0
    for f, nodes in faces_of.items():
        try:
            xyz, uv = gmsh.model.getClosestPoint(2, f, qm.X[nodes].ravel())
            n = np.array(gmsh.model.getNormal(f, uv)).reshape(-1, 3) * pa.face_cad_sign.get(f, 1)
            # reprojection exacte des nœuds sur la CAD (corrige l'erreur de corde des
            # surfaces composites) ; un déplacement anormal est ignoré
            xyz = np.array(xyz).reshape(-1, 3)
            mv = (~projected[nodes]) & (np.linalg.norm(xyz - qm.X[nodes], axis=1) < 0.5 * t_med)
            X_new[nodes[mv]] = xyz[mv]
            projected[nodes[mv]] = True
        except Exception:  # noqa: BLE001 - repli : normale du maillage
            n = _mesh_normals(qm, f, nodes)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        acc[nodes] += n
        cnt[nodes] += 1
        per_face_n.append((nodes, n))
    n_out = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-300)[:, None]
    mincos = np.ones(N)
    for nodes, n in per_face_n:
        mincos[nodes] = np.minimum(mincos[nodes], np.einsum("ij,ij->i", n_out[nodes], n))
    miter = 1.0 / np.clip(mincos, 1.0 / 3.0, 1.0)
    qm.X[:] = X_new
    return n_out, miter, faces_of


def _mesh_normals(qm: QuadMesh, f, nodes):
    q = qm.quads[qm.quad_face == f]
    X = qm.X
    n_el = np.cross(X[q[:, 2]] - X[q[:, 0]], X[q[:, 3]] - X[q[:, 1]])
    acc = np.zeros((len(X), 3))
    for k in range(4):
        np.add.at(acc, q[:, k], n_el)
    return acc[nodes]


def _node_adjacency(qm: QuadMesh):
    from scipy.sparse import coo_matrix
    Q = qm.quads
    N = len(qm.X)
    rows = np.concatenate([Q[:, k] for k in range(4)] * 2)
    cols = np.concatenate([Q[:, (k + 1) % 4] for k in range(4)] + [Q[:, (k - 1) % 4] for k in range(4)])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    A.data[:] = 1.0
    return A


def offset_nodes(pa, qm: QuadMesh, edge_nodes: np.ndarray | None = None) -> OffsetResult:
    n_out, miter, faces_of = node_normals(pa, qm)
    d = -n_out
    N = len(qm.X)
    t_med = pa.classification["t_median"] if pa.classification["t_median"] > 0 else 1.0
    t_acc = np.zeros(N)
    t_cnt = np.zeros(N)
    for f, nodes in faces_of.items():
        t_acc[nodes] += pa.face_thickness.get(f, t_med)
        t_cnt[nodes] += 1
    t_exp = np.where(t_cnt > 0, t_acc / np.maximum(t_cnt, 1), t_med)

    # rayons vers la peau opposée ET les chants (dessus de marche, congé de fond de
    # poche, biseau) : on ne retient que les surfaces de sortie de matière
    target = _Target(_load_surface(pa, list(pa.opp_faces) + list(pa.flank_faces)), pa.face_cad_sign)
    t_zone_max = max([t_med] + list(pa.thickness_zones) + [v for v in pa.face_thickness.values() if np.isfinite(v)])
    t_max = 1.6 * t_zone_max * float(miter.max())
    t_ray, j = cast_rays(target.st, qm.X, d, t_max, exit_cos=np.cos(np.radians(75.0)))
    hit = j >= 0
    face_hit = np.where(hit, target.st.face[np.maximum(j, 0)], -1)
    approx = qm.X + d * np.where(hit, t_ray, 0.0)[:, None]
    fb = ~hit
    if fb.any():
        # repli (rayon rasant en bord libre) : point le plus proche depuis un point
        # décalé d'une demi-épaisseur, donc encore dans la matière
        guess = qm.X[fb] + d[fb] * (0.5 * t_exp[fb])[:, None]
        p, ff, _ = target.nearest_cad(guess, d[fb])
        approx[fb] = p
        face_hit[fb] = ff

    X_top, resid = _cad_refine(approx, face_hit, t_exp)

    # nœuds sans impact (rayon rasant le long d'un chant en bord libre) : ils
    # reprennent le vecteur de décalage moyen de leurs voisins qui ont un impact
    # franc -> colonnes parallèles, jamais croisées ; puis reprojection sur la CAD.
    if fb.any() and len(qm.quads):
        nb = _node_adjacency(qm)
        good = ~fb
        D = X_top - qm.X
        todo = np.nonzero(fb)[0]
        for _ in range(6):
            if not len(todo):
                break
            acc = nb @ (D * good[:, None])
            cnt = nb @ good.astype(float)
            ok = cnt[todo] > 0
            sel = todo[ok]
            D[sel] = acc[sel] / cnt[sel][:, None]
            good[sel] = True
            todo = todo[~ok]
        sel = np.nonzero(fb & good)[0]
        if len(sel):
            p, ff, _ = target.nearest_cad(qm.X[sel] + D[sel], d[sel])
            X_top[sel], face_hit[sel], resid[sel] = p, ff, 0.0

    # démêlage local : colonnes qui se croisent (normales divergentes sur petites faces,
    # marches d'épaisseur, bords biseautés)
    X_top, face_hit, resid, n_untangle = _untangle(qm, d, X_top, face_hit, resid, target, t_exp)
    L = np.linalg.norm(X_top - qm.X, axis=1)
    lateral = np.linalg.norm((X_top - qm.X) - np.einsum("ij,ij->i", X_top - qm.X, d)[:, None] * d, axis=1)
    res = OffsetResult(X_top=X_top, direction=d, miter=miter, t_expected=t_exp,
                       t_ray=np.where(hit, t_ray, np.nan), reproj_err=resid / np.maximum(t_exp, 1e-12),
                       fallback=fb, top_face=face_hit)
    res.tilt = lateral / np.maximum(L, 1e-12)          # type: ignore[attr-defined]
    res.n_untangle_iter = n_untangle                   # type: ignore[attr-defined]
    return res


class _Target:
    """Cible (peau opposée + chants) : triangulation pour trouver les faces
    candidates, projection exacte sur la CAD pour le point final."""

    def __init__(self, st: SurfTri, signs: dict | None = None):
        self.st = st
        self.tree = cKDTree(st.centroid)
        self.signs = signs or {}

    def nearest_cad(self, pts: np.ndarray, dirs: np.ndarray, exit_cos: float = 0.26, k: int = 48):
        """Point le plus proche sur les faces CAD candidates (celles des k triangles
        les plus proches), en privilégiant les surfaces de sortie (normale · dir >
        exit_cos). Retourne (points, faces, distance)."""
        st = self.st
        k = min(k, len(st.T))
        _, J = self.tree.query(pts, k=k)
        J = J.reshape(len(pts), -1)
        cand = st.face[J]
        n = len(pts)
        best = [np.full(n, np.inf), np.zeros((n, 3)), np.zeros(n, np.int64)]
        anyb = [np.full(n, np.inf), np.zeros((n, 3)), np.zeros(n, np.int64)]
        for g in np.unique(cand):
            idx = np.nonzero((cand == g).any(axis=1))[0]
            try:
                xyz, uv = gmsh.model.getClosestPoint(2, int(g), pts[idx].ravel())
                xyz = np.array(xyz).reshape(-1, 3)
                nn = np.array(gmsh.model.getNormal(int(g), uv)).reshape(-1, 3) * self.signs.get(int(g), 1)
            except Exception:  # noqa: BLE001
                continue
            dd = np.linalg.norm(xyz - pts[idx], axis=1)
            for store, mask in ((anyb, np.ones(len(idx), bool)),
                                (best, np.einsum("ij,ij->i", nn, dirs[idx]) > exit_cos)):
                b = mask & (dd < store[0][idx])
                ii = idx[b]
                store[0][ii], store[1][ii], store[2][ii] = dd[b], xyz[b], g
        none = ~np.isfinite(best[0])
        for q in range(3):
            best[q][none] = anyb[q][none]
        miss = ~np.isfinite(best[0])
        best[1][miss] = pts[miss]
        return best[1], best[2], best[0]

    def nearest(self, pts: np.ndarray, dirs: np.ndarray | None = None, exit_cos: float = 0.26, k: int = 32):
        """Point le plus proche ; avec `dirs`, seulement parmi les triangles de sortie
        (normale sortante · direction > exit_cos) - repli sans filtre si aucun."""
        st = self.st
        k = min(k, len(st.T))
        _, J = self.tree.query(pts, k=k)
        J = J.reshape(len(pts), -1)
        best_d = np.full(len(pts), np.inf)
        best_p = pts.copy()
        best_j = np.zeros(len(pts), np.int64)
        any_d = np.full(len(pts), np.inf)
        any_p = pts.copy()
        any_j = np.zeros(len(pts), np.int64)
        for col in range(J.shape[1]):
            j = J[:, col]
            tri = st.T[j]
            q = _closest_on_triangles(pts, st.P[tri[:, 0]], st.P[tri[:, 1]], st.P[tri[:, 2]])
            dd = np.linalg.norm(q - pts, axis=1)
            b = dd < any_d
            any_d[b], any_p[b], any_j[b] = dd[b], q[b], j[b]
            if dirs is not None:
                ok = np.einsum("ij,ij->i", st.normal[j], dirs) > exit_cos
                b = ok & (dd < best_d)
                best_d[b], best_p[b], best_j[b] = dd[b], q[b], j[b]
        if dirs is None:
            return any_p, st.face[any_j]
        none = ~np.isfinite(best_d)
        best_p[none], best_j[none] = any_p[none], any_j[none]
        return best_p, st.face[best_j]


def _cad_refine(approx: np.ndarray, faces: np.ndarray, t_exp: np.ndarray):
    """Projection exacte sur la surface CAD de la face attribuée.

    Retourne (points, résidu) : le résidu est nul pour un point projeté ; si la
    projection saute de plus d'une demi-épaisseur (surface support hors découpe),
    on garde le point de la triangulation et le résidu vaut ce saut.
    """
    X = approx.copy()
    for g in np.unique(faces):
        m = faces == g
        try:
            xyz, _ = gmsh.model.getClosestPoint(2, int(g), approx[m].ravel())
            X[m] = np.array(xyz).reshape(-1, 3)
        except Exception:  # noqa: BLE001
            pass
    jump = np.linalg.norm(X - approx, axis=1)
    bad = jump > 0.5 * t_exp
    X[bad] = approx[bad]
    return X, np.where(bad, jump, 0.0)


def _oriented_quads(X: np.ndarray, Q: np.ndarray, d: np.ndarray) -> np.ndarray:
    Q = Q.copy()
    if len(Q):
        nq = np.cross(X[Q[:, 2]] - X[Q[:, 0]], X[Q[:, 3]] - X[Q[:, 1]])
        flip = np.einsum("ij,ij->i", nq, d[Q].mean(axis=1)) < 0
        Q[flip] = Q[flip][:, [0, 3, 2, 1]]
    return Q


def _untangle(qm: QuadMesh, d, X_top, faces, resid, target: _Target, t_exp,
              max_iter: int = 40, rings: int = 2):
    """Lissage local des vecteurs de décalage + reprojection sur la CAD, jusqu'à ce
    que les hexaèdres ne soient plus nettement moins bons que leur quad de base.

    Seules les zones fautives (et `rings` couronnes autour) bougent ; les nœuds
    restent sur la géométrie. On conserve le meilleur état rencontré.
    """
    from scipy.sparse import coo_matrix

    from .quality import HEX_CORNERS, scaled_jacobian
    from .repair import quad_quality

    Q = _oriented_quads(qm.X, qm.quads, d)
    N = len(qm.X)
    if len(Q) == 0:
        return X_top, faces, resid, 0
    rows = np.concatenate([Q[:, k] for k in range(4)] * 2)
    cols = np.concatenate([Q[:, (k + 1) % 4] for k in range(4)] + [Q[:, (k - 1) % 4] for k in range(4)])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    q2d = quad_quality(qm.X, Q)
    H = np.hstack([Q, Q + N])
    floor = np.minimum(0.3, 0.8 * q2d)

    def evaluate(top):
        sj = scaled_jacobian(np.vstack([qm.X, top]), H, HEX_CORNERS)
        bad = sj < floor
        return sj, bad

    def score(sj, bad):
        # priorité : aucun élément retourné, puis moins d'éléments médiocres, puis min
        return (int((sj <= 0).sum()), int(bad.sum()), -float(sj.min()))

    sj, bad = evaluate(X_top)
    best = (score(sj, bad), X_top.copy(), faces.copy(), resid.copy())
    it = 0
    for it in range(1, max_iter + 1):
        if not bad.any():
            break
        region = np.zeros(N, bool)
        region[np.unique(Q[bad])] = True
        for _ in range(rings):
            region |= (A @ region.astype(float)) > 0
        idx = np.nonzero(region)[0]
        D = X_top - qm.X
        Dn = (A @ D) / np.maximum(deg, 1)[:, None]
        Dnew = 0.5 * D[idx] + 0.5 * Dn[idx]
        xr, ff, _ = target.nearest_cad(qm.X[idx] + Dnew, d[idx])
        rr = np.zeros(len(idx))
        X_top = X_top.copy()
        faces = faces.copy()
        resid = resid.copy()
        X_top[idx], faces[idx], resid[idx] = xr, ff, rr
        sj, bad = evaluate(X_top)
        sc = score(sj, bad)
        if sc < best[0]:
            best = (sc, X_top.copy(), faces.copy(), resid.copy())
    return best[1], best[2], best[3], it
