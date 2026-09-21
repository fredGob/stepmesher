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


def _mesh_tet(pa, cfg, h, alg2d, alg3d, order, res, source: str):
    tc = cfg["tet"]
    gmsh.model.occ.importShapes(source)
    gmsh.model.occ.synchronize()
    if len(gmsh.model.getEntities(3)) == 0:
        raise RuntimeError("aucun solide")
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", max(h * tc["min_size_ratio"], cfg["mesh"]["min_size_mm"]))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(tc["curvature_n"]))
    gmsh.option.setNumber("Mesh.Algorithm", alg2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", alg3d)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    t1 = time.time()
    gmsh.model.mesh.generate(3)
    etype = 11 if order == 2 else 4
    n = sum(len(tg) for _, v in gmsh.model.getEntities(3)
            for et, tg in zip(*gmsh.model.mesh.getElements(3, v)[:2]) if et == etype)
    if n == 0:
        raise RuntimeError("aucun tétraèdre")
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


def run_tet_attempt(analysis_json: str, cfg_data: dict, out_prefix: str, source: int = 0, algo: int = 0) -> dict:
    """Un essai tétraédrique (une source géométrique, une combinaison d'algorithmes)."""
    t0 = time.time()
    out = Path(out_prefix)
    res = dict(kind="tet", passed=False, status="error", timings={})
    try:
        cfg = Config(cfg_data)
        pa = PartAnalysis.load(analysis_json)
        tc = cfg["tet"]
        order = int(tc["order"])
        h = tet_size(pa, cfg)
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
        if (low & ~imposed).any():
            reasons.append(f"qualité de forme min {q[~imposed].min():.3f} < {tc['min_shape_quality']} "
                           f"({int((low & ~imposed).sum())} élément(s) hors micro-géométrie)")
        m = dict(n_nodes=int(len(X)), n_elements=int(len(conn)), element_type="C3D10" if order == 2 else "C3D4",
                 size=h * hf, shape_quality=dict(min=float(q.min()), p05=float(np.percentile(q, 5)),
                                                 median=float(np.median(q))),
                 n_below_0_1=int((q < 0.1).sum()), n_negative_volume=n_neg, volume_mesh=float(vol.sum()),
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
