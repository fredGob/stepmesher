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
from ..mesh.repair import align_columns, fix_micro_edges, flip_repair, relax_boundary
from ..mesh.quad import (apply_strategy, even_boundaries, extract_reference_mesh, nodes_on_curves,
                         regular_expected_faces, setup_reference_model, skin_view)
from ..mesh.sizing import apply_sizing, base_size
from ..occ.loader import gmsh_session
from .recipe import Recipe


def reproject_moved(qm, X0: np.ndarray, tol: float = 1e-9) -> int:
    """Nœuds de la peau de référence déplacés depuis le maillage gmsh (lissages) : ramenés
    sur leur face CAD (gmsh.model.getClosestPoint, modèle de référence courant). Face d'un
    nœud = face d'un quad incident (celle du plus grand nombre de quads en cas de partage).
    Renvoie le nombre de nœuds reprojetés."""
    moved = np.flatnonzero(np.linalg.norm(qm.X - X0, axis=1) > tol)
    if not len(moved):
        return 0
    node_face = {}
    for q, f in zip(qm.quads, qm.quad_face):
        for n in q:
            node_face.setdefault(int(n), []).append(int(f))
    by_face: dict[int, list[int]] = {}
    for n in moved:
        fs = node_face.get(int(n))
        if fs:
            by_face.setdefault(max(set(fs), key=fs.count), []).append(int(n))
    n_done = 0
    for f, ns in by_face.items():
        try:
            c, _ = gmsh.model.getClosestPoint(2, f, qm.X[ns].ravel().tolist())
        except Exception:  # noqa: BLE001
            continue
        C = np.asarray(c, float).reshape(-1, 3)
        if len(C) == len(ns) and np.all(np.isfinite(C)):
            qm.X[ns] = C
            n_done += len(ns)
    return n_done


def reclassify_groups(pa, qm) -> int:
    """Quads et triangles d'une face fusionnée (tranches, pa.skin_face_groups), portant le numéro
    du représentant après retour à la numérotation de pa.brep : rendus à la tranche d'origine la
    plus proche de leur centre (triangulation d'analyse, sans extrapolation de surface). Les
    normales et la reprojection du décalage se font ensuite sur la bonne face CAO."""
    from ..mesh.offset import _load_surface, nearest_on_surface
    n = 0
    for rep, members in (getattr(pa, "skin_face_groups", None) or {}).items():
        st = _load_surface(pa, list(members))
        if not len(st.T):
            continue
        for arr, fa in ((qm.quads, qm.quad_face), (qm.tris, qm.tri_face)):
            sel = np.flatnonzero(fa == int(rep))
            if not len(sel):
                continue
            _, j = nearest_on_surface(st, qm.X[arr[sel]].mean(1))
            fa[sel] = st.face[j]
            n += len(sel)
    return n


def _recipe_in(rc: Recipe, o2n: dict | None) -> Recipe:
    """Recette dans la numérotation du modèle de peau (faces libres, algorithmes par face,
    fusions : numéros de pa.brep dans les recettes et les leviers)."""
    if not o2n:
        return rc
    d = rc.to_dict()
    d["free_faces"] = tuple(o2n[f] for f in rc.free_faces if f in o2n)
    d["face_alg"] = tuple((o2n[f], a) for f, a in rc.face_alg if f in o2n)
    d["merge"] = tuple((o2n[a], o2n[b]) for a, b in rc.merge if a in o2n and b in o2n)
    return Recipe.from_dict(d)


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
            # topologie virtuelle : tout le maillage de peau se fait dans la numérotation du
            # modèle de peau (pm, recette traduite rc_m) ; retour à celle de pa.brep avant le
            # décalage (modèle complet)
            pm, n2o, o2n = skin_view(pa)
            rc_m = _recipe_in(rc, o2n)
            res["virtual_topology"] = n2o is not None
            # stratégie (transfinis, fusion des micro-courbes) AVANT le champ de taille : les
            # courbes fusionnées n'imposent plus de nœud et ne doivent plus être raffinées
            res["strategy_info"] = apply_strategy(
                pm, rc_m, base_size(pa, cfg, rc), bool(cfg["mesh"].get("structured_patches", True)),
                float(cfg["mesh"].get("max_bend_angle_deg", 15.0)), float(cfg["mesh"].get("min_bend_size_frac", 0.0)),
                bool(cfg["mesh"].get("bend_angle_floor", False)), bool(cfg["mesh"].get("contour_arcs", False)),
                float(cfg["mesh"].get("large_face_elements", 0.0)), int(cfg["mesh"].get("large_face_alg", 6)),
                bool(cfg["mesh"].get("structured_harmonize", False)),
                float(cfg["mesh"].get("structured_patch_max_side_frac", 8.0)))
            res["sizing"] = apply_sizing(pm, cfg, rc_m, set(res["strategy_info"].get("merged_curves", [])),
                                         set(res["strategy_info"].get("structured_list", [])))
            t1 = time.time()
            if gmsh.option.getNumber("Mesh.RecombinationAlgorithm") in (2, 3):
                try:
                    res["parity_fixes"] = even_boundaries(
                        pm, set(res["strategy_info"].get("structured_list", [])) |
                        set(res["strategy_info"].get("structured_patch_list", [])))
                except Exception as e:  # noqa: BLE001
                    res["parity_fixes"] = f"erreur : {e}"[:200]
            try:
                gmsh.model.mesh.generate(2)
                res["gmsh_warning"] = None
            except Exception as e:  # noqa: BLE001
                # gmsh peut lever une erreur tout en ayant maillé la plupart des faces
                res["gmsh_warning"] = str(e)[:300]
            res["timings"]["quad"] = time.time() - t1
            qm = extract_reference_mesh(pm)
            X_gmsh = qm.X.copy()
            res["micro_edge_moves"] = fix_micro_edges(qm, cfg["mesh"]["micro_edge_ratio"] * cfg["mesh"]["min_size_mm"])
            if cfg["mesh"].get("align_columns", False) and len(qm.quads):
                # colonnes des faces transfinies redressées (patte à bouts en biais : colonnes
                # penchées sur toute la longueur) ; nœuds sur un sommet CAD fixes
                try:
                    lk0 = {int(t): i for i, t in enumerate(qm.node_tags)}
                    anchors = set()
                    for _, p_ in gmsh.model.getEntities(0):
                        tg, _, _ = gmsh.model.mesh.getNodes(0, p_)
                        anchors.update(lk0[int(t)] for t in tg if int(t) in lk0)
                    res["column_align"] = align_columns(
                        qm, list(res["strategy_info"].get("structured_list", [])) +
                        list(res["strategy_info"].get("structured_patch_list", [])), anchors)
                except Exception as e:  # noqa: BLE001
                    res["column_align"] = dict(error=f"{type(e).__name__}: {e}"[:200])
            if cfg["mesh"].get("boundary_relax", False) and len(qm.quads):
                # nœuds protégés : faces transfinies (plis, lanières, petites faces) et trous
                prot = np.zeros(len(qm.X), bool)
                lk = {int(t): i for i, t in enumerate(qm.node_tags)}
                for f in set(res["strategy_info"].get("structured_list", [])) | \
                        set(res["strategy_info"].get("structured_patch_list", [])):
                    try:
                        tg, _, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
                        prot[[lk[int(t)] for t in tg if int(t) in lk]] = True
                    except Exception:  # noqa: BLE001
                        pass
                prot[nodes_on_curves(qm, set(res["sizing"].get("hole_curves", [])))] = True
                # arcs du contour imposés : nœuds intérieurs fixes, extrémités libres (une facette
                # de 1 mm coincée entre deux arcs doit pouvoir s'élargir, part_014)
                for c in res["strategy_info"].get("contour_arc_list", []):
                    try:
                        tg, _, _ = gmsh.model.mesh.getNodes(1, c, includeBoundary=False)
                        prot[[lk[int(t)] for t in tg if int(t) in lk]] = True
                    except Exception:  # noqa: BLE001
                        pass
                q = cfg["quality"]
                res["boundary_relax"] = relax_boundary(
                    qm, prot, res["sizing"]["h0"], float(cfg["mesh"].get("boundary_relax_short_frac", 0.35)),
                    float(cfg["mesh"].get("boundary_relax_corner_deg", 25.0)),
                    int(cfg["mesh"].get("boundary_relax_window", 3)),
                    shape=dict(max_aspect_ratio=q["max_aspect_ratio"]))
            if cfg["mesh"]["local_repair"]:
                t_r = time.time()
                q = cfg["quality"]
                # faces transfinies : lissage seul, jamais de bascule (rangées régulières intactes)
                frozen = set(res["strategy_info"].get("structured_list", [])) | \
                    set(res["strategy_info"].get("structured_patch_list", []))
                res["quad_repairs"] = flip_repair(
                    qm, qm.fixed, threshold=cfg["mesh"].get("repair_threshold", 0.2),
                    shape=dict(max_angle_deg=q["max_angle_deg"], min_angle_deg=q["min_angle_deg"],
                               max_aspect_ratio=q["max_aspect_ratio"]), frozen_faces=frozen)
                res["timings"]["repair"] = time.time() - t_r
            # lissages (réparation, relaxation du contour) faits en 3D : sur une face courbe, la
            # moyenne des voisins tire le nœud vers la corde, dans la matière (part_021 : 4 nœuds
            # à 1,7 mm du pli R8 -> écart d'épaisseur 0,53) -> reprojection sur la face CAD
            res["reprojected_nodes"] = reproject_moved(qm, X_gmsh)
            # faces qui devraient être maillées en rangées régulières (4 coins) : contrôle
            # topologique (signalement) après la construction des SC8R
            try:
                expected_regular = regular_expected_faces(pm, res["sizing"]["h0"] / max(rc.size_mult, 1e-9))
            except Exception:  # noqa: BLE001
                expected_regular = {}
            meshed_faces = set(np.unique(np.r_[qm.quad_face, qm.tri_face]).tolist())
            missing = sorted(set(pm.ref_faces) - meshed_faces)
            if n2o is not None:
                missing = sorted(n2o[f] for f in missing)
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
            for f in pm.ref_faces:
                try:
                    per = sum(gmsh.model.occ.getMass(1, abs(c)) for _, c in gmsh.model.getBoundary([(2, f)], oriented=False))
                    face_width[int(f)] = 2 * gmsh.model.occ.getMass(2, f) / max(per, 1e-30)
                except Exception:  # noqa: BLE001
                    pass
            hole_curves = set(res["sizing"].get("hole_curves", []))
            free_c, hole_c = classify_curves(pm, hole_curves)
            ns_free = nodes_on_curves(qm, free_c)
            ns_hole = nodes_on_curves(qm, hole_c)
            if n2o is not None:
                # retour à la numérotation de pa.brep (décalage, qualité, rapports, leviers)
                tr = np.vectorize(lambda f: n2o.get(int(f), int(f)), otypes=[np.int64])
                if len(qm.quad_face):
                    qm.quad_face = tr(qm.quad_face)
                if len(qm.tri_face):
                    qm.tri_face = tr(qm.tri_face)
                struct_nodes = {n2o[f]: v for f, v in struct_nodes.items()}
                face_width = {n2o[f]: v for f, v in face_width.items()}
                expected_regular = {n2o[f]: v for f, v in expected_regular.items()}
                si = res["strategy_info"]
                grp = getattr(pa, "skin_face_groups", None) or {}
                for k in ("structured_list", "structured_strips", "structured_patch_list", "structured_dropped",
                          "large_faces"):
                    if isinstance(si.get(k), list):
                        # face fusionnée -> toutes ses tranches (exemptions de régularité des plis)
                        si[k] = [t for f in si[k] for t in grp.get(n2o.get(int(f), int(f)), [n2o.get(int(f), int(f))])]
                res["reclassified_elements"] = reclassify_groups(pa, qm)

            # géométrie complète pour les normales CAD et la peau opposée
            gmsh.model.add("full")
            gmsh.model.occ.importShapes(pa.brep)
            gmsh.model.occ.synchronize()
            t2 = time.time()
            off = offset_nodes(pa, qm, np.union1d(ns_free, ns_hole))
            # nœud du contour déplacé par la relaxation dont le décalage dévie (chanfrein, chant
            # incliné à son nouvel emplacement, part_029 : 1,34 mm pour 2,71) : remis à sa
            # position gmsh (sommet CAD), décalage recalculé
            if pa.classification.get("thickness_kind", pa.kind) == "constant" and len(ns_free):
                slid = np.zeros(len(qm.X), bool)
                slid[ns_free] = np.linalg.norm(qm.X[ns_free] - X_gmsh[ns_free], axis=1) > 1e-9
                if slid.any():
                    Lr = np.linalg.norm(off.X_top - qm.X, axis=1)
                    dev = np.abs(Lr / np.maximum(off.t_expected * off.miter, 1e-12) - 1.0)
                    back = slid & (dev > 0.8 * float(cfg["quality"]["max_thickness_deviation"]))
                    if back.any():
                        qm.X[back] = X_gmsh[back]
                        off = offset_nodes(pa, qm, np.union1d(ns_free, ns_hole))
                        res["relax_reverted"] = int(back.sum())
            res["timings"]["offset"] = time.time() - t2
        sm = build_solid_shell(qm, off, pa.obb_axes[0])
        quality_kind = pa.classification.get("thickness_kind", pa.kind)
        m, soft = quality.evaluate(sm, off, cfg, quality_kind)
        # régularité visuelle : taille NOMINALE (sans size_mult), plis transfinis exemptés
        struct_bends = {f for b in pa.bends for f in b["faces"]} & set(res["strategy_info"].get("structured_list", []))
        m["regularity"] = quality.regularity(
            sm, res["sizing"]["h0"] / max(rc.size_mult, 1e-9), struct_bends,
            float(cfg["quality"].get("regularity_jump_max", 1.5)), float(cfg["quality"].get("regularity_small_frac", 0.5)))
        # régularité topologique (étoiles dans une face à 4 coins) : signalée, non bloquante ;
        # entre dans la pénalité de départage des recettes qui passent
        try:
            topo = quality.topology(qm.quads, qm.quad_face, ~qm.fixed, qm.X, expected_regular)
            m["topology"] = topo
            w_topo = float(cfg["quality"].get("regularity_topology_weight", 1.0))
            m["regularity"]["topology_pct"] = topo["pct"]
            m["regularity"]["penalty"] = m["regularity"]["penalty"] + w_topo * topo["pct"]
        except Exception as e:  # noqa: BLE001
            m["topology"] = dict(error=f"{type(e).__name__}: {e}"[:200])
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
