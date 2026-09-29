"""Repli tétraédrique (C3D10 par défaut, C3D4 en option).

Déclenché pour une pièce massive, ou quand toutes les recettes SC8R ont échoué
(jonctions en T, nervures, chapes épaisses : géométries qu'une peau décalée ne
peut pas représenter). Exécuté en sous-processus isolé comme les essais SC8R.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import gmsh
import numpy as np

from ..analyze.pipeline import PartAnalysis
from ..config import Config
from ..occ.loader import gmsh_session
from .quality import mesh_pieces

# arêtes des nœuds milieux C3D10 (Abaqus) : 5:(1,2) 6:(2,3) 7:(3,1) 8:(1,4) 9:(2,4) 10:(3,4)
ABQ_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
TET_FACES = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]


def tet_size(pa: PartAnalysis, cfg: Config) -> float:
    t = cfg["tet"]
    tm = pa.classification["t_median"] if pa.classification.get("t_median", 0) > 0 else 0.0
    h = t["size_frac"] * pa.diag
    if tm > 0:
        h = min(h, t["thickness_factor"] * tm)
    return float(min(max(h, t["min_size_mm"]), t["max_size_mm"]))


def corner_quality(X: np.ndarray, T4: np.ndarray):
    """Volume signé et qualité de forme (rayon inscrit normalisé, 1 = régulier) des tétras."""
    a, b, c, d = (X[T4[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    areas = sum(0.5 * np.linalg.norm(np.cross(X[T4[:, j]] - X[T4[:, i]], X[T4[:, k]] - X[T4[:, i]]), axis=1)
                for i, j, k in TET_FACES)
    r_in = 3 * np.abs(vol) / np.maximum(areas, 1e-300)
    edges = [np.linalg.norm(X[T4[:, j]] - X[T4[:, i]], axis=1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    lmax = np.max(np.stack(edges, 1), axis=1)
    # tétra régulier : r_in / lmax = 1 / (2*sqrt(6))
    q = r_in / np.maximum(lmax, 1e-300) * 2 * np.sqrt(6)
    return vol, q


def _abaqus_order(X: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Réordonne les nœuds milieux gmsh selon la convention Abaqus C3D10 (par géométrie)."""
    e = conn[0]
    perm = [0, 1, 2, 3]
    for i, j in ABQ_EDGES:
        mid = 0.5 * (X[e[i]] + X[e[j]])
        k = 4 + int(np.argmin([np.linalg.norm(X[e[4 + m]] - mid) for m in range(6)]))
        perm.append(k)
    if sorted(perm) != list(range(10)):
        raise RuntimeError("ordre des nœuds C3D10 non reconnu")
    return conn[:, perm]


def fillet_bands(h: float, max_angle_deg: float, min_frac: float, max_aspect: float = 2.5) -> list[int]:
    """Congés (face à 4 côtés, une courbure principale ~1/R, l'autre ~0) maillés en
    bande régulière transfinie : 1 élément par max_angle_deg en travers (sans descendre
    sous min_frac * h), taille uniforme le long (<= max_aspect x taille en travers).
    Les courbes partagées entre deux congés gardent le premier découpage imposé."""
    from .quad import _bend_sides, _split_count
    fixed: dict[int, int] = {}
    done = []
    for _, f in gmsh.model.getEntities(2):
        try:
            lo, hi = gmsh.model.getParametrizationBounds(2, f)
            k1, k2, _, _ = gmsh.model.getPrincipalCurvatures(f, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
            kmax, kmin = max(abs(k1[0]), abs(k2[0])), min(abs(k1[0]), abs(k2[0]))
            if kmax < 1e-6 or kmin > 0.05 * kmax:
                continue
            r = _bend_sides(f)
        except Exception:  # noqa: BLE001
            continue
        if r is None:
            continue
        sides, lens, corners = r
        arcs = (sum(lens[1]), sum(lens[3]))
        gen = (lens[0][0], lens[2][0])
        R = 1.0 / kmax
        # vrai congé simple : 4 courbes, arcs de même longueur (sinon extrémités coupées
        # en biais : maillage 3D en échec), plus courts que les génératrices, angle <= 180°
        if sum(len(x) for x in sides) != 4 or max(arcs) > 1.2 * min(arcs) or max(arcs) > max(gen) \
                or max(arcs) / R > np.pi * 1.05:
            continue
        arc = max(arcs)
        n_arc = int(np.ceil(np.degrees(arc / R) / max_angle_deg - 1e-6))
        n_arc = max(1, min(n_arc, int(arc / (min_frac * h)) if min_frac > 0 else n_arc))
        # une corde unique sur un arc > 45° déforme les faces voisines (coins toriques)
        if np.degrees(arc / R) > 45:
            n_arc = max(n_arc, 2)
        n_arc = max(n_arc, len(sides[1]), len(sides[3]))
        h_gen = min(h, max_aspect * arc / n_arc)
        n_gen = max(1, int(round(max(gen) / h_gen)))
        want = {}
        for side, ln, tot in ((sides[0], lens[0], n_gen), (sides[1], lens[1], n_arc),
                              (sides[2], lens[2], n_gen), (sides[3], lens[3], n_arc)):
            for c, k in zip(side, _split_count(ln, tot)):
                want[c] = k + 1
        if any(c in fixed and fixed[c] != n for c, n in want.items()):
            continue
        try:
            for c, n in want.items():
                gmsh.model.mesh.setTransfiniteCurve(c, n)
            if sum(len(x) for x in sides) == 4:
                gmsh.model.mesh.setTransfiniteSurface(f)
            else:
                gmsh.model.mesh.setTransfiniteSurface(f, cornerTags=corners)
        except Exception:  # noqa: BLE001
            continue
        fixed.update(want)
        done.append(f)
    return done


def short_curve_field(h: float, hmin: float, frac: float) -> int:
    """Champ de taille : ~longueur de la courbe près des courbes < frac x h (nœuds de congés,
    petits chanfreins), retour à h sur 2 h. Sans lui, le tétra s'y écrase (part_020)."""
    fl = []
    for _, c in gmsh.model.getEntities(1):
        L = gmsh.model.occ.getMass(1, c)
        if L >= frac * h:
            continue
        d = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d, "CurvesList", [c])
        th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(th, "InField", d)
        gmsh.model.mesh.field.setNumber(th, "SizeMin", max(L, hmin))
        gmsh.model.mesh.field.setNumber(th, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(th, "DistMin", 0.5 * L)
        gmsh.model.mesh.field.setNumber(th, "DistMax", 2 * h)
        fl.append(th)
    if fl:
        mn = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mn, "FieldsList", fl)
        gmsh.model.mesh.field.setAsBackgroundMesh(mn)
    return len(fl)


def _mesh_tet(pa, cfg, h, alg2d, alg3d, order, res, source: str):
    tc = cfg["tet"]
    gmsh.model.occ.importShapes(source)
    gmsh.model.occ.synchronize()
    if len(gmsh.model.getEntities(3)) == 0:
        raise RuntimeError("aucun solide")
    hmin = max(h * tc["min_size_ratio"], cfg["mesh"]["min_size_mm"])
    frac = float(tc.get("short_curve_frac", 0.0))
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(tc["curvature_n"]))
    gmsh.option.setNumber("Mesh.Algorithm", alg2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", alg3d)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    bands = []
    if tc.get("fillet_bands", True):
        bands = fillet_bands(h, float(tc.get("fillet_max_angle_deg", 30.0)),
                             float(tc.get("fillet_min_size_frac", 0.35)))
    res["fillet_bands"] = len(bands)
    res["short_curves_refined"] = short_curve_field(h, hmin, frac) if frac > 0 else 0
    t1 = time.time()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:  # noqa: BLE001
        if not bands:
            raise
        # bandes de congés incompatibles avec le mailleur 3D : même essai sans bandes
        res["fillet_bands"] = 0
        res["fillet_bands_error"] = str(e)[:200]
        gmsh.model.remove()
        gmsh.model.add("tet")
        gmsh.model.occ.importShapes(source)
        gmsh.model.occ.synchronize()
        if frac > 0:
            short_curve_field(h, hmin, frac)
        gmsh.model.mesh.generate(3)
    etype = 11 if order == 2 else 4
    n = sum(len(tg) for _, v in gmsh.model.getEntities(3)
            for et, tg in zip(*gmsh.model.mesh.getElements(3, v)[:2]) if et == etype)
    if n == 0:
        raise RuntimeError("aucun tétraèdre" + (f" (avec bandes de congés : {res['fillet_bands_error']})"
                                                if res.get("fillet_bands_error") else ""))
    res["midnodes"] = "sur la CAD" if order == 2 else "-"
    if order == 2:
        # validité des éléments courbes ; sinon nœuds milieux sur les arêtes droites
        tq = [t for _, v in gmsh.model.getEntities(3)
              for et, tg in zip(*gmsh.model.mesh.getElements(3, v)[:2]) if et == etype for t in tg]
        sjq = np.array(gmsh.model.mesh.getElementQualities(tq, "minSJ")) if tq else np.zeros(0)
        res["curved_invalid"] = int((sjq <= 0).sum())
        if (sjq <= 0).any():
            gmsh.model.mesh.setOrder(1)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            gmsh.model.mesh.setOrder(2)
            res["midnodes"] = "arêtes droites (éléments courbes invalides)"
    res["timings"]["mesh"] = time.time() - t1


def _extract(order: int):
    """Nœuds, connectivité (ordre Abaqus), nœuds de peau du maillage gmsh courant."""
    etype = 11 if order == 2 else 4
    tags, coords, _ = gmsh.model.mesh.getNodes()
    idx = np.full(int(tags.max()) + 1, -1, np.int64)
    idx[tags.astype(np.int64)] = np.arange(len(tags))
    X = coords.reshape(-1, 3)
    conns = []
    for _, v in gmsh.model.getEntities(3):
        ets, _, ens = gmsh.model.mesh.getElements(3, v)
        for et, en in zip(ets, ens):
            if et == etype:
                conns.append(idx[en.astype(np.int64)].reshape(-1, 10 if order == 2 else 4))
    if not conns:
        raise RuntimeError("aucun tétraèdre produit")
    conn = np.concatenate(conns)
    used = np.unique(conn)
    remap = np.full(len(X), -1, np.int64)
    remap[used] = np.arange(len(used))
    X, conn = X[used], remap[conn]
    skin = set()
    for _, f in gmsh.model.getEntities(2):
        st, _, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
        skin.update(int(t) for t in st)
    skin_idx = remap[idx[np.array(sorted(skin), dtype=np.int64)]]
    skin_idx = skin_idx[skin_idx >= 0]
    if order == 2:
        conn = _abaqus_order(X, conn)
    vol, q = corner_quality(X, conn[:, :4])
    if (vol <= 0).all():            # orientation globale inversée
        conn[:, [1, 2]] = conn[:, [2, 1]]
        if order == 2:
            conn[:, [4, 6, 8, 9]] = conn[:, [6, 4, 9, 8]]
        vol, q = corner_quality(X, conn[:, :4])
    return X, conn, skin_idx, vol, q


TET_ALGOS = ((6, 1, 1.0), (1, 1, 1.0), (6, 10, 1.0))


def tet_sources(pa) -> list[tuple[str, str, float | None]]:
    """Géométrie préparée (trous bouchés, micro-arêtes supprimées), puis STEP d'origine
    avec correction de micro-arêtes plus douce, puis STEP brut : une tolérance trop
    agressive peut rendre le maillage de surface auto-intersectant."""
    return [("préparée", pa.brep, None)] + \
           [(f"STEP, micro-arêtes {t} mm", pa.source, t) for t in (0.1, 0.05, 0.02, 0.01, 0.005)] + \
           [("STEP brut", pa.source, None)]


def run_tet_attempt(analysis_json: str, cfg_data: dict, out_prefix: str, source: int = 0, algo: int = 0,
                    size_mult: float = 1.0) -> dict:
    """Un essai tétraédrique (une source géométrique, une combinaison d'algorithmes)."""
    t0 = time.time()
    out = Path(out_prefix)
    res = dict(kind="tet", passed=False, status="error", timings={})
    try:
        cfg = Config(cfg_data)
        pa = PartAnalysis.load(analysis_json)
        tc = cfg["tet"]
        order = int(tc["order"])
        h = tet_size(pa, cfg) * size_mult
        name, src, fix = tet_sources(pa)[source]
        alg2d, alg3d, hf = TET_ALGOS[algo]
        res["size"] = h * hf
        res["algorithms"] = dict(geometry=name, alg2d=alg2d, alg3d=alg3d, size_factor=hf)
        with gmsh_session(cfg):
            if fix:
                gmsh.option.setNumber("Geometry.Tolerance", fix)
                gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
                gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
            _mesh_tet(pa, cfg, h * hf, alg2d, alg3d, order, res, src)
            X, conn, skin_idx, vol, q = _extract(order)
        n_neg = int((vol <= 0).sum())
        # arête < 10 % de la taille min de maille : imposée par une micro-géométrie CAD
        hmin = max(h * hf * tc["min_size_ratio"], cfg["mesh"]["min_size_mm"])
        C4 = conn[:, :4]
        emin = np.min(np.stack([np.linalg.norm(X[C4[:, j]] - X[C4[:, i]], axis=1)
                                for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], 1), axis=1)
        imposed = emin < 0.1 * hmin
        low = q < tc["min_shape_quality"]
        reasons = []
        if n_neg:
            reasons.append(f"{n_neg} tétraèdre(s) de volume négatif")
        n_pieces = mesh_pieces(len(X), conn[:, :4])
        if n_pieces > 1:
            reasons.append(f"maillage en {n_pieces} morceaux disjoints (critère dur)")
        if (low & ~imposed).any():
            reasons.append(f"qualité de forme min {q[~imposed].min():.3f} < {tc['min_shape_quality']} "
                           f"({int((low & ~imposed).sum())} élément(s) hors micro-géométrie)")
        m = dict(n_nodes=int(len(X)), n_elements=int(len(conn)), element_type="C3D10" if order == 2 else "C3D4",
                 size=h * hf, shape_quality=dict(min=float(q.min()), p05=float(np.percentile(q, 5)),
                                                 median=float(np.median(q))),
                 n_below_0_1=int((q < 0.1).sum()), n_negative_volume=n_neg, n_pieces=n_pieces, volume_mesh=float(vol.sum()),
                 geometry_imposed=dict(count=int((low & imposed).sum()), edge_threshold_mm=0.1 * hmin,
                                       elements=(np.nonzero(low & imposed)[0] + 1).tolist()[:500]),
                 volume_cad=float(pa.import_info["final"]["volume"]), geometry=name)
        m["volume_rel_error"] = abs(m["volume_mesh"] - m["volume_cad"]) / max(m["volume_cad"], 1e-30)
        m["score"] = float(-n_neg + (q[~imposed].min() if (~imposed).any() else 1.0))
        res.update(metrics=m, passed=not reasons, reasons=reasons, status="passed" if not reasons else "failed")
        np.savez_compressed(out.with_suffix(".npz"), nodes=X, conn=conn, skin=skin_idx, quality=q)
        res["mesh_file"] = str(out.with_suffix(".npz"))
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["reasons"] = [f"{type(e).__name__}: {e}"]
        res["traceback"] = traceback.format_exc()[-2000:]
    res["timings"]["total"] = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    return res
