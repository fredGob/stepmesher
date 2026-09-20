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


def apply_sizing(pa, cfg, recipe) -> dict:
    m = cfg["mesh"]
    t_med = pa.classification["t_median"] if pa.classification["t_median"] > 0 else 1.0
    h0 = cfg.target_size(pa.diag, t_med) * recipe.size_mult
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
    hb_list = []
    for b in pa.bends:
        arc = b["radius"] * math.radians(b["angle_deg"])
        hb = max(min(arc / recipe.n_per_bend, h0), hmin)
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
    short = {c: clen(c) for c in ref_curves if clen(c) < 0.5 * h0}
    for lo, hi in ((0.0, 0.2 * h0), (0.2 * h0, 0.5 * h0)):
        cs = [c for c, L in short.items() if lo <= L < hi]
        if cs:
            smin = max(min(short[c] for c in cs), hmin)
            d = _distance_to_curves(cs, 40)
            fields.append(_threshold(d, smin, h0, 0.0, 4 * h0))
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
