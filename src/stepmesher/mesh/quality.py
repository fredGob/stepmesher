"""Métriques de qualité et verdict binaire (seuils configurables)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .hexa import SolidShellMesh
from .offset import OffsetResult

# voisins de chaque coin, ordre donnant un jacobien positif pour un hexa valide
HEX_CORNERS = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
               (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
WEDGE_CORNERS = [(0, 1, 2, 3), (1, 2, 0, 4), (2, 0, 1, 5), (3, 5, 4, 0), (4, 3, 5, 1), (5, 4, 3, 2)]
HEX_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def scaled_jacobian(X: np.ndarray, conn: np.ndarray, corners) -> np.ndarray:
    """Jacobien normalisé minimal par élément (1 = parfait, <= 0 = retourné)."""
    if len(conn) == 0:
        return np.zeros(0)
    out = np.full(len(conn), np.inf)
    for c, a, b, d in corners:
        e1 = X[conn[:, a]] - X[conn[:, c]]
        e2 = X[conn[:, b]] - X[conn[:, c]]
        e3 = X[conn[:, d]] - X[conn[:, c]]
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        nrm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
        out = np.minimum(out, det / np.maximum(nrm, 1e-300))
    return out


def hex_internal_jacobian(X: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Déterminant minimal du mapping trilineaire dans chaque hexaèdre.

    Les jacobiens aux huit coins ne détectent pas forcément une inversion interne
    d'un SC8R gauchi. La grille $3x3x3$ couvre aussi le centre d'intégration.
    """
    if len(conn) == 0:
        return np.zeros(0)
    points = (-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5))
    P = X[conn]
    out = np.full(len(conn), np.inf)
    for xi in points:
        for eta in points:
            for zeta in points:
                s = HEX_SIGNS
                dxi = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
                deta = 0.125 * (1 + s[:, 0] * xi) * s[:, 1] * (1 + s[:, 2] * zeta)
                dzeta = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) * s[:, 2]
                J = np.stack((np.einsum("i,nij->nj", dxi, P),
                              np.einsum("i,nij->nj", deta, P),
                              np.einsum("i,nij->nj", dzeta, P)), axis=1)
                out = np.minimum(out, np.linalg.det(J))
    return out


def quad_metrics(X: np.ndarray, Q: np.ndarray):
    """Élancement (arête max / min), angles internes, gauchissement (deg)."""
    if len(Q) == 0:
        z = np.zeros(0)
        return z, z, z, z
    P = [X[Q[:, k]] for k in range(4)]
    edges = [P[(k + 1) % 4] - P[k] for k in range(4)]
    lens = np.stack([np.linalg.norm(e, axis=1) for e in edges], 1)
    aspect = lens.max(1) / np.maximum(lens.min(1), 1e-300)
    angs = []
    for k in range(4):
        u = -edges[(k - 1) % 4]
        v = edges[k]
        c = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    angs = np.stack(angs, 1)

    def tri_n(a, b, c):
        n = np.cross(b - a, c - a)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    w1 = np.einsum("ij,ij->i", tri_n(P[0], P[1], P[2]), tri_n(P[0], P[2], P[3]))
    w2 = np.einsum("ij,ij->i", tri_n(P[1], P[2], P[3]), tri_n(P[1], P[3], P[0]))
    warp = np.degrees(np.arccos(np.clip(np.minimum(w1, w2), -1, 1)))
    return aspect, angs.min(1), angs.max(1), warp


def mesh_pieces(n_nodes: int, *conns) -> int:
    """Nombre de morceaux disjoints (éléments reliés par au moins un nœud commun)."""
    conns = [np.asarray(c) for c in conns if len(c)]
    if not conns:
        return 0
    rows = np.concatenate([np.repeat(np.arange(len(c)) + sum(len(d) for d in conns[:i]), c.shape[1])
                           for i, c in enumerate(conns)])
    cols = np.concatenate([c.ravel() for c in conns])
    n_el = sum(len(c) for c in conns)
    A = coo_matrix((np.ones(len(rows)), (rows, cols + n_el)), shape=(n_el + n_nodes,) * 2)
    used = np.zeros(n_el + n_nodes, bool)
    used[:n_el] = True
    used[cols + n_el] = True
    _, lab = connected_components(A, directed=False)
    return int(len(np.unique(lab[used])))


def mesh_cracks(X: np.ndarray, quads, tris=None, tol: float = 0.05, min_path: float = 1.0,
                top: int = 10) -> dict:
    """Discontinuités d'un maillage surfacique (peau de référence) : fissures et arêtes
    non manifold. Critère DUR, indépendant de la CAD.

    Fissure = nœud de bord (arête libre) situé à moins de `tol` d'une autre arête libre qu'on
    ne peut pas atteindre en suivant le bord sur moins de `min_path` mm : deux bords libres
    superposés, là où les faces auraient dû partager leurs nœuds. Cas d'origine : echelle
    part_021, passe B (micro-arêtes supprimées à l'import gmsh) : deux faces de peau décousues,
    fente de 16 mm, 19 nœuds doublés à 0,004 mm, un seul morceau (mesh_pieces ne voit rien).
    Le chemin minimal le long du bord écarte les replis d'arêtes microscopiques (déjà rejetés
    par le plancher absolu de taille)."""
    import heapq
    from scipy.spatial import cKDTree
    Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    T = np.asarray(tris if tris is not None else np.zeros((0, 3)), dtype=np.int64).reshape(-1, 3)
    E = np.concatenate([np.stack([F, np.roll(F, -1, 1)], 2).reshape(-1, 2) for F in (Q, T) if len(F)]
                       or [np.zeros((0, 2), np.int64)])
    out = dict(n_nodes=0, n_nonmanifold_edges=0, positions=[])
    if len(E) == 0:
        return out
    E = np.sort(E, 1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    out["n_nonmanifold_edges"] = int((cnt > 2).sum())
    B = uniq[cnt == 1]
    if len(B) == 0:
        return out
    Lb = np.linalg.norm(X[B[:, 1]] - X[B[:, 0]], axis=1)
    adj: dict[int, list] = {}
    for (a, b), L in zip(B.tolist(), Lb.tolist()):
        adj.setdefault(a, []).append((b, L))
        adj.setdefault(b, []).append((a, L))
    bn = np.array(sorted(adj))
    tree = cKDTree(X[bn])
    A, Bp = X[B[:, 0]], X[B[:, 1]]
    mid = 0.5 * (A + Bp)

    def near_along(n, a, b):
        """a ou b atteint depuis n en suivant le bord sur moins de min_path mm."""
        dist, heap = {n: 0.0}, [(0.0, n)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in (a, b):
                return True
            if d > dist.get(u, np.inf):
                continue
            for v, L in adj[u]:
                nd = d + L
                if nd < min_path and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return False

    crack = set()
    for j in range(len(B)):
        a, b = int(B[j, 0]), int(B[j, 1])
        ab = Bp[j] - A[j]
        den = max(float(ab @ ab), 1e-300)
        for k in tree.query_ball_point(mid[j], 0.5 * Lb[j] + tol):
            n = int(bn[k])
            if n in (a, b) or n in crack:
                continue
            s = min(max(float((X[n] - A[j]) @ ab) / den, 0.0), 1.0)
            if np.linalg.norm(A[j] + s * ab - X[n]) < tol and not near_along(n, a, b):
                crack.add(n)
    out["n_nodes"] = len(crack)
    if crack:
        P = X[sorted(crack)]
        out["positions"] = np.round(P[:top], 2).tolist()
    return out


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(min=None, p05=None, median=None, p95=None, max=None)
    return dict(min=float(x.min()), p05=float(np.percentile(x, 5)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def evaluate(sm: SolidShellMesh, off: OffsetResult, cfg, kind: str):
    """Retourne (métriques + verdict, masque des éléments hors cibles)."""
    q = cfg["quality"]
    X = sm.nodes
    sj_h = scaled_jacobian(X, sm.hexa, HEX_CORNERS)
    sj_w = scaled_jacobian(X, sm.wedge, WEDGE_CORNERS)
    sj = np.r_[sj_h, sj_w]
    det_h = hex_internal_jacobian(X, sm.hexa)
    Qb = sm.hexa[:, :4]
    aspect, amin, amax, warp = quad_metrics(X, Qb)
    n_el = len(sm.hexa) + len(sm.wedge)
    tri_pct = 100.0 * len(sm.wedge) / max(n_el, 1)

    # épaisseur par nœud vs attendue (onglet compris)
    L = np.linalg.norm(off.X_top - X[:sm.n_ref], axis=1)
    t_dev = np.abs(L / np.maximum(off.t_expected * off.miter, 1e-12) - 1.0)
    # éléments à cheval sur une marche d'épaisseur (écart entre nœuds d'un même élément)
    Lh = L[sm.hexa[:, :4]] if len(sm.hexa) else np.zeros((0, 4))
    step = (Lh.max(1) / np.maximum(Lh.min(1), 1e-12) - 1.0) if len(Lh) else np.zeros(0)
    n_step = int(np.sum(step > cfg["analysis"]["constant_thickness_rel_tol"]))

    m = dict(
        n_nodes=int(len(X)), n_elements=int(n_el), n_sc8r=int(len(sm.hexa)), n_sc6r=int(len(sm.wedge)),
        triangle_pct=tri_pct, scaled_jacobian=_stats(sj), internal_jacobian=_stats(det_h),
        n_negative_jacobian=int(np.sum((sj_h <= 0) | (det_h <= 0)) + np.sum(sj_w <= 0)),
        aspect_ratio=_stats(aspect), min_angle=_stats(amin), max_angle=_stats(amax), warp_deg=_stats(warp),
        reprojection_error=_stats(off.reproj_err), thickness_deviation=_stats(t_dev),
        offset_tilt=_stats(getattr(off, "tilt", np.zeros(0))),
        untangle_iterations=int(getattr(off, "n_untangle_iter", 0)),
        element_thickness=_stats(sm.thickness), n_fallback_nodes=int(off.fallback.sum()),
        n_step_elements=n_step, elements_through_thickness=1,
    )
    # quad de base avec une arête < 10 % de la taille min de maille : imposé par une
    # micro-géométrie CAD (exempté des seuils de forme, jamais du critère « non retourné »)
    Pq = X[Qb]
    emin_h = np.min(np.linalg.norm(np.roll(Pq, -1, axis=1) - Pq, axis=2), axis=1) if len(Qb) else np.zeros(0)
    Pw = X[sm.wedge[:, :3]] if len(sm.wedge) else np.zeros((0, 3, 3))
    emin_w = np.min(np.linalg.norm(np.roll(Pw, -1, axis=1) - Pw, axis=2), axis=1) if len(Pw) else np.zeros(0)
    emin = np.r_[emin_h, emin_w]
    imposed = emin < cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"]
    m["geometry_imposed"] = dict(count=int((imposed & (sj < q["min_scaled_jacobian"])).sum()),
                                 edge_threshold_mm=cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"])
    m["min_edge"] = _stats(emin)
    reasons = []
    # --- critères durs (100 % des éléments) ---
    if n_el == 0:
        reasons.append("aucun élément")
    if m["n_negative_jacobian"]:
        reasons.append(f"{m['n_negative_jacobian']} élément(s) retourné(s) (jacobien <= 0)")
    # une pièce = un seul morceau de maillage (peau de référence coupée par un chant, etc.)
    m["n_pieces"] = mesh_pieces(len(X), sm.hexa, sm.wedge)
    if m["n_pieces"] > 1:
        reasons.append(f"maillage en {m['n_pieces']} morceaux disjoints (critère dur)")
    # continuité : faces voisines qui ne partagent pas leurs nœuds (fissure, bords superposés)
    m["cracks"] = mesh_cracks(X, Qb, sm.wedge[:, :3] if len(sm.wedge) else None,
                              float(q.get("crack_tol_mm", 0.05)), float(q.get("crack_min_path_mm", 1.0)))
    n_crack = m["cracks"]["n_nodes"] + m["cracks"]["n_nonmanifold_edges"]
    if m["cracks"]["n_nodes"]:
        reasons.append(f"maillage DISCONTINU : {m['cracks']['n_nodes']} nœud(s) sur une fissure (bords libres "
                       f"superposés à < {q.get('crack_tol_mm', 0.05)} mm, faces non raccordées ; critère dur), "
                       f"vers {m['cracks']['positions'][0]}")
    if m["cracks"]["n_nonmanifold_edges"]:
        reasons.append(f"{m['cracks']['n_nonmanifold_edges']} arête(s) partagée(s) par plus de 2 éléments "
                       f"(maillage superposé, critère dur)")
    # plancher absolu de taille : jamais exempté, même pour une arête « imposée par la CAD »
    # (sinon un défaut STEP non nettoyé produit des éléments dégénérés qui passent quand même)
    min_edge_mm = q.get("min_absolute_edge_mm", 0.0)
    too_small = emin < min_edge_mm if min_edge_mm > 0 else np.zeros(n_el, bool)
    if too_small.any():
        reasons.append(f"{int(too_small.sum())} élément(s) avec arête < {min_edge_mm} mm "
                       f"(critère dur, plancher absolu, min observé {emin[too_small].min():.4g} mm)")
    internal_bad = det_h <= 0
    hard_bad = ((sj < q["hard_min_scaled_jacobian"]) & ~imposed) | too_small
    hard_bad[:len(det_h)] |= internal_bad
    if (hard_bad & ~too_small).any():
        reasons.append(f"jacobien normalisé min {sj[hard_bad & ~too_small].min():.3f} < {q['hard_min_scaled_jacobian']} "
                       f"(critère dur, {int((hard_bad & ~too_small).sum())} élément(s))")
    if tri_pct > q["allow_wedge_pct"] + 1e-12:
        reasons.append(f"{tri_pct:.2f} % de triangles > {q['allow_wedge_pct']} % toléré")
    if off.reproj_err.max(initial=0) > q["max_reprojection_error"]:
        reasons.append(f"erreur de reprojection max {off.reproj_err.max():.3f} > {q['max_reprojection_error']}")
    if kind == "constant" and t_dev.max(initial=0) > q["max_thickness_deviation"]:
        reasons.append(f"écart d'épaisseur max {t_dev.max():.3f} > {q['max_thickness_deviation']}")
    # --- critères cibles (tolérance en % d'éléments) ---
    nh = len(sm.hexa)
    soft = np.zeros(n_el, bool)
    detail = {}
    for name, bad in (("jacobien < cible", sj < q["min_scaled_jacobian"]),
                      ("élancement", np.r_[aspect > q["max_aspect_ratio"], np.zeros(n_el - nh, bool)]),
                      ("angle min", np.r_[amin < q["min_angle_deg"], np.zeros(n_el - nh, bool)]),
                      ("angle max", np.r_[amax > q["max_angle_deg"], np.zeros(n_el - nh, bool)]),
                      ("gauchissement", np.r_[warp > q["max_warp_deg"], np.zeros(n_el - nh, bool)])):
        if bad.any():
            detail[name] = int(bad.sum())
        soft |= bad & ~imposed
    soft_all = soft | (imposed & (sj < q["min_scaled_jacobian"]))
    n_soft = int(soft.sum())
    pct = 100.0 * n_soft / max(n_el, 1)
    # tôle constante : tolérance stricte ; épaisseur variable (rampes) : tolérance plus
    # large (la transition lissée impose quelques éléments en biais). Critères durs inchangés.
    soft_limit = q["soft_violation_pct"]
    if kind == "variable":
        soft_limit = q.get("soft_violation_pct_variable", soft_limit)
    # petits maillages : le % ne tolérerait aucun élément imposé par la CAD
    min_count = int(q.get("soft_violation_min_count", 0))
    m["soft_violations"] = dict(count=n_soft, pct=pct, limit_pct=soft_limit, min_count=min_count,
                                by_criterion=detail, elements=(np.nonzero(soft)[0] + 1).tolist()[:500])
    if pct > soft_limit + 1e-12 and n_soft > min_count:
        reasons.append(f"{n_soft} élément(s) hors cibles ({pct:.2f} % > {soft_limit} % et > {min_count}) : {detail}")
    m["passed"] = not reasons
    m["reasons"] = reasons
    # score pour départager des essais tous en échec (le moins mauvais)
    # score (plus grand = meilleur) : pondère les défauts par gravité
    m["n_hard_bad"] = int(hard_bad.sum())
    m["score"] = float(-(1000 * m["n_negative_jacobian"] + 1000 * max(0, m.get("n_pieces", 1) - 1)
                         + (1000 + 10 * n_crack if n_crack else 0) + 20 * m["n_hard_bad"] + 100 * tri_pct + 10 * pct)
                       + (sj.min() if len(sj) else -1))
    return m, soft_all


def regularity(sm: SolidShellMesh, h0: float, exempt_faces=frozenset(), jump_max: float = 1.5,
               small_frac: float = 0.5) -> dict:
    """Régularité visuelle de la peau de référence (critère « esthétique », non bloquant).

    Taille d'un quad = racine de son aire. Hors faces exemptées (plis transfinis : petits
    par construction, 3 éléments dans le rayon) :
    - small_pct : % de quads plus petits que small_frac x h0 (h0 = taille nominale) ;
    - jump_pct  : % des arêtes partagées entre deux quads dont le rapport de tailles
      dépasse jump_max (règle de Fred : transitions <= 1,5) ;
    - penalty   : small_pct + jump_pct, départage les recettes qui passent (plus petit =
      plus régulier). Le % hors cibles favorisait les maillages fins (défauts dilués)."""
    Q = sm.hexa[:, :4]
    if len(Q) == 0 or h0 <= 0:
        return dict(small_pct=0.0, jump_pct=0.0, jump_p95=1.0, penalty=0.0, n_counted=0)
    P = sm.nodes[Q]
    size = np.sqrt(0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1))
    keep = ~np.isin(sm.hexa_face, np.fromiter(exempt_faces, int, len(exempt_faces)))
    small_pct = 100.0 * float(np.mean(size[keep] < small_frac * h0)) if keep.any() else 0.0
    # arêtes partagées entre deux quads comptés
    e = np.sort(np.stack([Q, np.roll(Q, -1, 1)], 2).reshape(-1, 2), 1)
    owner = np.repeat(np.arange(len(Q)), 4)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = owner[:-1][same], owner[1:][same]
    both = keep[a] & keep[b]
    a, b = a[both], b[both]
    ratio = np.maximum(size[a], size[b]) / np.maximum(np.minimum(size[a], size[b]), 1e-300)
    jump_pct = 100.0 * float(np.mean(ratio > jump_max)) if len(ratio) else 0.0
    return dict(small_pct=small_pct, jump_pct=jump_pct,
                jump_p95=float(np.percentile(ratio, 95)) if len(ratio) else 1.0,
                penalty=small_pct + jump_pct, n_counted=int(keep.sum()))


def topology(quads: np.ndarray, quad_face: np.ndarray, interior: np.ndarray, X: np.ndarray,
             expected: dict, min_irregular: int = 2, top: int = 20) -> dict:
    """Régularité TOPOLOGIQUE du maillage de peau (signalement, non bloquant).

    Nœud irrégulier = nœud intérieur à une face CAD (pas sur une courbe) touché par 3, 5 ou
    plus de quads au lieu de 4 : les « étoiles » d'un maillage libre. Inévitables sur une
    face de forme libre, elles sont un DÉFAUT sur une face qui devrait être maillée en
    rangées régulières (`expected` : 4 coins, côtés opposés voisins -> patte, lanière, pli,
    rectangle ; quad.regular_expected_faces). Cas d'origine : upper part_001, pattes de
    7,4 m x 36 mm maillées en libre, 154 étoiles, invisibles des critères par élément.

    Renvoie : irregular_pct (tous nœuds intérieurs), faces fautives (>= min_irregular nœuds
    irréguliers dans une face attendue régulière) avec position, n_faces, n_irregular (dans
    ces faces) et pct (en % des nœuds intérieurs, sert au départage des recettes)."""
    Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    N = len(X)
    if len(Q) == 0:
        return dict(irregular_pct=0.0, n_faces=0, n_irregular=0, pct=0.0, faces=[])
    val = np.bincount(Q.ravel(), minlength=N)
    face_of = np.full(N, -1, np.int64)
    face_of[Q.ravel()] = np.repeat(np.asarray(quad_face, np.int64), 4)
    inner = np.asarray(interior, bool) & (val > 0)
    irr = inner & (val != 4)
    n_in = int(inner.sum())
    faces = []
    for f, info in expected.items():
        m = irr & (face_of == int(f))
        k = int(m.sum())
        if k >= min_irregular:
            n_f = int((inner & (face_of == int(f))).sum())
            faces.append(dict(face=int(f), n_irregular=k, n_interior=n_f,
                              valence_3=int((m & (val == 3)).sum()), valence_5plus=int((m & (val >= 5)).sum()),
                              centre=np.round(X[m].mean(0), 1).tolist(), sides_mm=info.get("sides")))
    faces.sort(key=lambda d: -d["n_irregular"])
    n_bad = int(sum(d["n_irregular"] for d in faces))
    return dict(irregular_pct=100.0 * float(irr.sum()) / max(n_in, 1), n_faces=len(faces), n_irregular=n_bad,
                pct=100.0 * n_bad / max(n_in, 1), n_expected_faces=len(expected), faces=faces[:top])
