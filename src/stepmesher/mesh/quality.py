"""Métriques de qualité et verdict binaire (seuils configurables)."""
from __future__ import annotations

import numpy as np

from .hexa import SolidShellMesh
from .offset import OffsetResult

# voisins de chaque coin, ordre donnant un jacobien positif pour un hexa valide
HEX_CORNERS = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
               (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
WEDGE_CORNERS = [(0, 1, 2, 3), (1, 2, 0, 4), (2, 0, 1, 5), (3, 5, 4, 0), (4, 3, 5, 1), (5, 4, 3, 2)]


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
        triangle_pct=tri_pct, scaled_jacobian=_stats(sj), n_negative_jacobian=int(np.sum(sj <= 0)),
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
    emin = np.min(np.linalg.norm(np.roll(Pq, -1, axis=1) - Pq, axis=2), axis=1) if len(Qb) else np.zeros(0)
    imposed = np.r_[emin < cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"], np.zeros(n_el - len(Qb), bool)]
    m["geometry_imposed"] = dict(count=int((imposed & (sj < q["min_scaled_jacobian"])).sum()),
                                 edge_threshold_mm=cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"])
    reasons = []
    # --- critères durs (100 % des éléments) ---
    if n_el == 0:
        reasons.append("aucun élément")
    if m["n_negative_jacobian"]:
        reasons.append(f"{m['n_negative_jacobian']} élément(s) retourné(s) (jacobien <= 0)")
    hard_bad = (sj < q["hard_min_scaled_jacobian"]) & ~imposed
    if hard_bad.any():
        reasons.append(f"jacobien normalisé min {sj[hard_bad].min():.3f} < {q['hard_min_scaled_jacobian']} "
                       f"(critère dur, {int(hard_bad.sum())} élément(s))")
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
    m["soft_violations"] = dict(count=n_soft, pct=pct, by_criterion=detail,
                                elements=(np.nonzero(soft)[0] + 1).tolist()[:500])
    if pct > q["soft_violation_pct"] + 1e-12:
        reasons.append(f"{n_soft} élément(s) hors cibles ({pct:.2f} % > {q['soft_violation_pct']} %) : {detail}")
    m["passed"] = not reasons
    m["reasons"] = reasons
    # score pour départager des essais tous en échec (le moins mauvais)
    # score (plus grand = meilleur) : pondère les défauts par gravité
    m["n_hard_bad"] = int(hard_bad.sum())
    m["score"] = float(-(1000 * m["n_negative_jacobian"] + 20 * m["n_hard_bad"] + 100 * tri_pct + 10 * pct)
                       + (sj.min() if len(sj) else -1))
    return m, soft_all
