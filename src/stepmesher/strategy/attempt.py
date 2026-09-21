"""Un essai de maillage complet, exécuté dans un sous-processus isolé.

peau de référence -> quads -> décalage -> SC8R -> qualité. Le résultat est écrit
sur disque (JSON + npz) : un plantage de gmsh ne perd que cet essai.
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
from ..mesh import quality
from ..mesh.hexa import build_solid_shell
from ..mesh.offset import offset_nodes
from ..mesh.repair import fix_micro_edges, flip_repair
from ..mesh.quad import apply_strategy, extract_reference_mesh, nodes_on_curves, setup_reference_model
from ..mesh.sizing import apply_sizing
from ..occ.loader import gmsh_session
from .recipe import Recipe


def classify_curves(pa: PartAnalysis, hole_curves: set[int]) -> tuple[set[int], set[int]]:
    """Courbes de bord de la peau de référence : bords libres / bords de trous."""
    ref = set(pa.ref_faces)
    ref_curves = {c for f in ref for c in pa.face_curves.get(f, [])}
    flank_curves = {c for f in pa.flank_faces for c in pa.face_curves.get(f, [])}
    border = ref_curves & flank_curves
    return border - hole_curves, border & hole_curves


def run_attempt(analysis_json: str, recipe: dict, cfg_data: dict, out_prefix: str) -> dict:
    t0 = time.time()
    out = Path(out_prefix)
    res = dict(recipe=recipe, passed=False, status="error", timings={})
    try:
        cfg = Config(cfg_data)
        pa = PartAnalysis.load(analysis_json)
        rc = Recipe.from_dict(recipe)
        with gmsh_session(cfg):
            setup_reference_model(pa)
            res["sizing"] = apply_sizing(pa, cfg, rc)
            res["strategy_info"] = apply_strategy(pa, rc, res["sizing"]["h0"], bool(cfg["mesh"].get("structured_patches", True)))
            t1 = time.time()
            try:
                gmsh.model.mesh.generate(2)
                res["gmsh_warning"] = None
            except Exception as e:  # noqa: BLE001
                # gmsh peut lever une erreur tout en ayant maillé la plupart des faces
                res["gmsh_warning"] = str(e)[:300]
            res["timings"]["quad"] = time.time() - t1
            qm = extract_reference_mesh(pa)
            res["micro_edge_moves"] = fix_micro_edges(qm, cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"])
            if cfg["mesh"]["local_repair"]:
                t_r = time.time()
                res["quad_repairs"] = flip_repair(qm, qm.fixed, threshold=cfg["mesh"].get("repair_threshold", 0.2))
                res["timings"]["repair"] = time.time() - t_r
            meshed_faces = set(np.unique(np.r_[qm.quad_face, qm.tri_face]).tolist())
            missing = sorted(set(pa.ref_faces) - meshed_faces)
            res["faces_not_meshed"] = missing
            if qm.n_elems == 0:
                res["status"] = "failed"
                res["reasons"] = ["aucun élément produit" + (f" ({res['gmsh_warning']})" if res["gmsh_warning"] else "")]
                return _finish(res, out, t0)
            if missing:
                res["status"] = "failed"
                res["reasons"] = [f"{len(missing)} face(s) de référence non maillée(s) : {missing[:10]}"]
                return _finish(res, out, t0)
            # nœuds (indices locaux) de chaque pli structuré, pour la recette adaptative
            lookup = {int(t): i for i, t in enumerate(qm.node_tags)}
            struct_nodes = {}
            for f in res["strategy_info"].get("structured_list", []):
                try:
                    tg, _, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
                    struct_nodes[int(f)] = {lookup[int(t)] for t in tg if int(t) in lookup}
                except Exception:  # noqa: BLE001
                    pass
            face_width = {}
            for f in pa.ref_faces:
                try:
                    per = sum(gmsh.model.occ.getMass(1, abs(c)) for _, c in gmsh.model.getBoundary([(2, f)], oriented=False))
                    face_width[int(f)] = 2 * gmsh.model.occ.getMass(2, f) / max(per, 1e-30)
                except Exception:  # noqa: BLE001
                    pass
            hole_curves = set(res["sizing"].get("hole_curves", []))
            free_c, hole_c = classify_curves(pa, hole_curves)
            ns_free = nodes_on_curves(qm, free_c)
            ns_hole = nodes_on_curves(qm, hole_c)

            # géométrie complète pour les normales CAD et la peau opposée
            gmsh.model.add("full")
            gmsh.model.occ.importShapes(pa.brep)
            gmsh.model.occ.synchronize()
            t2 = time.time()
            off = offset_nodes(pa, qm, np.union1d(ns_free, ns_hole))
            res["timings"]["offset"] = time.time() - t2
        sm = build_solid_shell(qm, off, pa.obb_axes[0])
        m, soft = quality.evaluate(sm, off, cfg, pa.kind)
        res["metrics"] = m
        hf_all = np.r_[sm.hexa_face, sm.wedge_face]
        sj_h = quality.scaled_jacobian(sm.nodes, sm.hexa, quality.HEX_CORNERS)
        sj_w = quality.scaled_jacobian(sm.nodes, sm.wedge, quality.WEDGE_CORNERS)
        bad_el = soft | (np.r_[sj_h, sj_w] < cfg["quality"]["hard_min_scaled_jacobian"])
        bad_el[:len(sm.hexa)] |= quality.hex_internal_jacobian(sm.nodes, sm.hexa) <= 0
        bad_el[len(sm.hexa):] = True          # triangles restants
        res["bad_faces"] = sorted({int(f) for f in hf_all[bad_el]})
        bad_nodes = set(np.unique(np.r_[sm.hexa[bad_el[:len(sm.hexa)], :4].ravel(),
                                        sm.wedge[bad_el[len(sm.hexa):], :3].ravel()]).tolist())
        res["implicated_structured"] = sorted(f for f, ns in struct_nodes.items() if ns & bad_nodes)
        # faces fautives étroites (largeur moyenne 2A/P < 0.5 h0) : voisine la plus tangente
        h0 = res["sizing"]["h0"]
        ref = set(pa.ref_faces)
        merges = []
        for f in res["bad_faces"]:
            if face_width.get(f, 1e30) < 0.5 * h0:
                cands = [(ang, b if a == f else a) for a, b, ang in pa.face_adjacency
                         if f in (a, b) and (b if a == f else a) in ref and ang < 15.0]
                if cands:
                    merges.append((int(f), int(min(cands)[1])))
        res["sliver_merges"] = merges
        res["passed"] = m["passed"]
        res["reasons"] = m["reasons"]
        res["status"] = "passed" if m["passed"] else "failed"
        sj = np.r_[sj_h, sj_w]
        np.savez_compressed(
            out.with_suffix(".npz"), nodes=sm.nodes, n_ref=sm.n_ref, hexa=sm.hexa, wedge=sm.wedge,
            hexa_face=sm.hexa_face, wedge_face=sm.wedge_face, normal=sm.normal, stack=sm.stack,
            axis1=sm.axis1, axis2=sm.axis2, thickness=sm.thickness, sj=sj, ns_free=ns_free, ns_hole=ns_hole,
            reproj=off.reproj_err, fallback=off.fallback, t_expected=off.t_expected, soft=soft)
        res["mesh_file"] = str(out.with_suffix(".npz"))
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["reasons"] = [f"{type(e).__name__}: {e}"]
        res["traceback"] = traceback.format_exc()[-2000:]
    return _finish(res, out, t0)


def _finish(res, out: Path, t0) -> dict:
    res["timings"]["total"] = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=_default))
    return res


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))
