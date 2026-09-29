"""Champ de taille gmsh : taille cible relative + raffinements locaux.

h0 = clamp(min(frac*diag, k*t), min, max) * size_mult, puis :
- par face : k * épaisseur locale (pièces à épaisseur variable) ;
- plis : n_per_bend éléments sur l'arc (distance aux lignes de pli) ;
- trous conservés : n_per_hole éléments sur le périmètre ;
- arêtes CAD courtes : gradation douce au lieu d'un saut de taille brutal.
"""
from __future__ import annotations

import math

import gmsh


def _threshold(dist_field: int, smin: float, smax: float, dmin: float, dmax: float) -> int:
    f = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(f, "SizeMin", smin)
    gmsh.model.mesh.field.setNumber(f, "SizeMax", smax)
    gmsh.model.mesh.field.setNumber(f, "DistMin", dmin)
    gmsh.model.mesh.field.setNumber(f, "DistMax", dmax)
    return f


def _distance_to_curves(curves, sampling: int) -> int:
    f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f, "CurvesList", sorted(curves))
    gmsh.model.mesh.field.setNumber(f, "Sampling", int(max(20, min(sampling, 4000))))
    return f


def base_size(pa, cfg, recipe) -> float:
    t_med = pa.classification["t_median"] if pa.classification["t_median"] > 0 else 1.0
    return cfg.target_size(pa.diag, t_med) * recipe.size_mult


def apply_sizing(pa, cfg, recipe, skip_curves=frozenset(), structured=frozenset()) -> dict:
    """skip_curves : courbes fusionnées en composite (quad.merge_micro_curves) : leur sommet
    commun n'impose plus de nœud, inutile (et nuisible) de raffiner autour.
    structured : faces déjà maillées en transfini (quad.apply_strategy) : un pli dont toutes
    les faces y sont n'a pas besoin de champ de taille."""
    m = cfg["mesh"]
    t_med = pa.classification["t_median"] if pa.classification["t_median"] > 0 else 1.0
    h0 = base_size(pa, cfg, recipe)
    hmin = m["min_size_mm"]
    fields, info = [], dict(h0=h0)
    ref = set(pa.ref_faces)
    curve_len = {}

    def clen(c):
        if c not in curve_len:
            curve_len[c] = gmsh.model.occ.getMass(1, c)
        return curve_len[c]

    # --- taille par face (épaisseur locale) ---
    per_face = {}
    for f in ref:
        t = pa.face_thickness.get(f, t_med)
        hf = cfg.target_size(pa.diag, t) * recipe.size_mult
        if hf < 0.9 * h0:
            per_face.setdefault(round(hf, 3), []).append(f)
    for hf, faces in per_face.items():
        fc = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumbers(fc, "SurfacesList", faces)
        gmsh.model.mesh.field.setNumber(fc, "VIn", hf)
        gmsh.model.mesh.field.setNumber(fc, "VOut", 1e22)
        gmsh.model.mesh.field.setNumber(fc, "IncludeBoundary", 1)
        fields.append(fc)
    info["face_sizes"] = {str(k): len(v) for k, v in per_face.items()}

    # --- plis ---
    # un pli transfini a ses nœuds imposés : le champ ne servirait qu'à raffiner les faces
    # VOISINES (taille du pli jusqu'à ~2 h0 de ses génératrices, alors que celles-ci portent
    # des nœuds espacés de ~h0) -> rosaces le long du pli, ailes étroites entièrement fines
    # (part_004 face 5 : 992 éléments de 1,1 mm ; part_014). Champ gardé pour les plis libres.
    hb_list, n_bend_skipped = [], 0
    for b in pa.bends:
        if b["faces"] and set(b["faces"]) <= set(structured):
            n_bend_skipped += 1
            continue
        arc = b["radius"] * math.radians(b["angle_deg"])
        # même règle que le transfini du pli (quad.structured_bends)
        n_ang = math.ceil(b["angle_deg"] / max(m.get("max_bend_angle_deg", 30.0), 1e-6) - 1e-6)
        n_arc = max(recipe.n_per_bend, n_ang)
        hb_floor = m.get("min_bend_size_frac", 0.0) * h0
        if m.get("bend_angle_floor", False):
            hb_floor = min(hb_floor, arc / max(n_ang, 1))
        hb = max(min(arc / n_arc, h0), hmin, hb_floor)
        if hb >= 0.95 * h0:
            continue
        curves = {c for f in b["faces"] for c in pa.face_curves.get(f, [])}
        if not curves:
            continue
        L = max(clen(c) for c in curves)
        d = _distance_to_curves(curves, int(2 * L / hb))
        fields.append(_threshold(d, hb, h0, 0.6 * arc, 0.6 * arc + 2 * h0))
        hb_list.append(hb)
    info["bend_sizes"] = hb_list
    info["bends_structured_no_field"] = n_bend_skipped

    # --- trous conservés ---
    hole_curves = set()
    for h in pa.holes_kept:
        wall = set(h["faces"])
        curves = {c for f in ref for c in pa.face_curves.get(f, []) if any(c in pa.face_curves.get(w, []) for w in wall)}
        if not curves:
            continue
        hole_curves |= curves
        hh = max(min(math.pi * h["diameter"] / recipe.n_per_hole, h0), hmin)
        if hh < 0.95 * h0:
            d = _distance_to_curves(curves, int(math.pi * h["diameter"] / hh * 2))
            fields.append(_threshold(d, hh, h0, 0.0, max(h["diameter"], 2 * h0)))
    info["hole_curves"] = sorted(hole_curves)

    # --- arêtes courtes : gradation ---
    ref_curves = {c for f in ref for c in pa.face_curves.get(f, [])}
    short = {c: clen(c) for c in ref_curves if c not in skip_curves and clen(c) < 0.5 * h0}
    # plancher : une arête courte impose ses nœuds, mais le champ ne descend pas sous
    # short_curve_min_size_frac x h0 (sinon des anneaux concentriques d'éléments minuscules
    # autour d'une seule arête de 1 mm) ; portée short_curve_dist_frac x h0 (4 x h0 avant :
    # rosaces aux coins arrondis du contour)
    floor = m.get("short_curve_min_size_frac", 0.0) * h0
    reach = m.get("short_curve_dist_frac", 4.0) * h0
    for lo, hi in ((0.0, 0.2 * h0), (0.2 * h0, 0.5 * h0)):
        cs = [c for c, L in short.items() if lo <= L < hi]
        if cs and floor < 0.95 * h0:
            smin = max(min(short[c] for c in cs), hmin, floor)
            d = _distance_to_curves(cs, 40)
            fields.append(_threshold(d, smin, h0, 0.0, reach))
    info["short_curves"] = len(short)

    if fields:
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(m.get("curvature_n", 0)))
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    return info
