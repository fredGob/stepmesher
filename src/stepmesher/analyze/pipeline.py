"""Étape 0/1 : import, analyse, bouchage des petits trous, géométrie préparée.

Produit un `PartAnalysis` sérialisable (JSON + npz) et un fichier `prepared.brep`.
Les essais de maillage réimportent ce .brep : la numérotation des faces y est
déterministe, les tags de l'analyse restent donc valables dans chaque essai.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from ..occ.loader import aabb_diag, geom_state, import_step
from ..occ.topology import SurfTri, build_analysis_mesh, obb
from . import classify as C
from .features import detect_bends, detect_holes, free_edge_length
from ..occ.holes import fill_holes
from .fingerprint import exact_hash, exact_invariants, feature_vector
from .skins import SkinAnalysis, analyze_skins

log = logging.getLogger(__name__)


@dataclass
class PartAnalysis:
    source: str
    brep: str
    tri_file: str
    import_info: dict
    invariants: dict
    exact_hash: str
    features: list[float]
    diag: float
    obb_dims: list[float]
    obb_axes: list[list[float]]
    kind: str
    classification: dict
    reference_side: int
    reference_reason: str
    ref_faces: list[int]
    opp_faces: list[int]
    flank_faces: list[int]
    failed_faces: list[int]
    face_thickness: dict
    face_cad_sign: dict                  # normale sortante = signe * getNormal
    face_curves: dict                    # face -> courbes du bord
    face_adjacency: list                 # [fa, fb, angle dièdre (deg)]
    thickness_zones: list[float]
    bends: list[dict]
    sharp_edges: list[dict]
    holes_kept: list[dict]
    holes_filled: list[dict]
    holes_fill_failed: list[dict]
    free_edge_length: float
    timings: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["face_thickness"] = {str(k): v for k, v in self.face_thickness.items()}
        d["face_cad_sign"] = {str(k): v for k, v in self.face_cad_sign.items()}
        d["face_curves"] = {str(k): v for k, v in self.face_curves.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "PartAnalysis":
        d = dict(d)
        d["face_thickness"] = {int(k): v for k, v in d["face_thickness"].items()}
        d["face_cad_sign"] = {int(k): v for k, v in d["face_cad_sign"].items()}
        d["face_curves"] = {int(k): v for k, v in d["face_curves"].items()}
        return cls(**d)

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_json(), indent=1, default=_json_default))

    @classmethod
    def load(cls, path: Path) -> "PartAnalysis":
        return cls.from_json(json.loads(Path(path).read_text()))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def cad_normal_signs(st: SurfTri, n_per_face: int = 3) -> dict[int, int]:
    """Signe liant la normale CAD gmsh (getNormal) à la normale sortante, par face."""
    signs = {}
    order = np.argsort(st.face, kind="stable")
    fs = st.face[order]
    starts = np.r_[0, np.nonzero(fs[1:] != fs[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(fs)]
    for s, e in zip(starts, ends):
        f = int(fs[s])
        idx = order[s:e]
        idx = idx[np.argsort(-st.area[idx])[:n_per_face]]
        try:
            _, uv = gmsh.model.getClosestPoint(2, f, st.centroid[idx].ravel())
            n = np.array(gmsh.model.getNormal(f, uv)).reshape(-1, 3)
            v = float(np.sum(np.einsum("ij,ij->i", n, st.normal[idx])))
            signs[f] = 1 if v >= 0 else -1
        except Exception:  # noqa: BLE001
            signs[f] = 1
    return signs


def exact_dihedrals(adjacency: dict, face_curves: dict, signs: dict, max_mesh_angle: float = 45.0) -> dict:
    """Angle dièdre exact (normales CAD) le long de l'arête commune la plus longue.

    La triangulation d'analyse surestime l'angle aux raccords tangents (normales
    prises au centre de triangles grossiers) ; on corrige par la CAD pour les
    paires candidates (angle maillage < max_mesh_angle).
    """
    out = {}
    for (a, b), v in adjacency.items():
        if v["angle_deg"] >= max_mesh_angle:
            continue
        common = set(face_curves.get(a, ())) & set(face_curves.get(b, ()))
        if not common:
            continue
        try:
            c = max(common, key=lambda c: gmsh.model.occ.getMass(1, c))
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            us = lo[0] + (hi[0] - lo[0]) * np.array([0.2, 0.5, 0.8])
            pts = np.array(gmsh.model.getValue(1, c, us))
            ns = []
            for f in (a, b):
                _, uv = gmsh.model.getClosestPoint(2, f, pts)
                ns.append(np.array(gmsh.model.getNormal(f, uv)).reshape(-1, 3) * signs.get(f, 1))
            cosang = np.clip(np.einsum("ij,ij->i", ns[0], ns[1]), -1, 1)
            out[(a, b)] = float(np.degrees(np.arccos(cosang)).max())
        except Exception:  # noqa: BLE001 - on garde l'angle maillage
            pass
    return out


def refine_bends_cad(bends: list[dict], st: SurfTri) -> None:
    """Rayon exact (courbure principale CAD) et angle = aire / (longueur * rayon)."""
    for b in bends:
        radii, area = [], 0.0
        for f in b["faces"]:
            m = st.face == f
            if not m.any():
                continue
            area += float(st.area[m].sum())
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(-st.area[idx])[:3]]
            try:
                _, uv = gmsh.model.getClosestPoint(2, f, st.centroid[idx].ravel())
                k1, k2, _, _ = gmsh.model.getPrincipalCurvatures(f, uv)
                k = np.maximum(np.abs(np.asarray(k1)), np.abs(np.asarray(k2)))
                radii += [1.0 / x for x in k if x > 1e-9]
            except Exception:  # noqa: BLE001
                pass
        if radii and b["length"] > 0:
            R = float(np.median(radii))
            b["radius_mesh_estimate"] = b["radius"]
            b["radius"] = R
            b["angle_deg"] = float(min(np.degrees(area / (b["length"] * R)), 180.0))


def _face_areas_seq() -> list[float]:
    return [gmsh.model.occ.getMass(2, t) for _, t in gmsh.model.getEntities(2)]


def _analyze_current(cfg, diag_hint: float | None = None):
    a = cfg["analysis"]
    gs = geom_state()
    diag = diag_hint or aabb_diag()
    t0 = time.time()
    st = build_analysis_mesh(diag * a["sample_size_frac"], a["sample_curvature"])
    t1 = time.time()
    t_est = 2 * gs.volume / max(gs.area, 1e-30)
    t_max = float(min(a["max_thickness_mm"], max(4 * t_est, 1e-3 * diag)))
    sk = analyze_skins(st, cfg, t_max)
    t2 = time.time()
    return st, sk, dict(triangulation=t1 - t0, rays=t2 - t1), t_max


def prepare_part(step_path: str | Path, workdir: str | Path, cfg, reference_skin: str | None = None) -> PartAnalysis:
    step_path, workdir = Path(step_path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    msgs: list[str] = []
    timings = {}
    T0 = time.time()
    ir = import_step(str(step_path), cfg)
    msgs += ir.messages
    timings["import"] = time.time() - T0
    inv = exact_invariants()
    ehash = exact_hash(inv)

    st, sk, tt, t_max = _analyze_current(cfg)
    timings.update({f"{k}_1": v for k, v in tt.items()})
    P_obb_w = st.area
    _, axes, dims = obb(st.centroid, P_obb_w)
    diag = float(np.linalg.norm(dims))

    # --- trous : bouchage des petits trous ---
    t_med = sk.t_median if np.isfinite(sk.t_median) else 1.0
    holes = detect_holes(st, sk, t_med)
    thr = cfg.hole_threshold(diag)
    to_fill = [h for h in holes if h["is_hole"] and h["diameter"] <= thr]
    t0 = time.time()

    def reimport():
        gmsh.model.occ.remove(gmsh.model.occ.getEntities(), recursive=True)
        gmsh.model.occ.synchronize()
        import_step(str(step_path), cfg)   # même import + nettoyage qu'au départ

    filled, fill_failed, method = fill_holes(to_fill, reimport)
    if to_fill:
        msgs.append(f"{len(filled)} trou(s) bouché(s) par {method}" +
                    (f", {len(fill_failed)} en échec" if fill_failed else ""))
    timings["hole_filling"] = time.time() - t0
    holes_kept = [h for h in holes if h["is_hole"] and h not in to_fill]

    # --- géométrie préparée : écriture puis relecture (tags déterministes) ---
    brep = workdir / "prepared.brep"
    seq_before = _face_areas_seq()
    gmsh.write(str(brep))
    gmsh.model.occ.remove(gmsh.model.occ.getEntities(), recursive=True)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.importShapes(str(brep))
    gmsh.model.occ.synchronize()
    seq_after = _face_areas_seq()
    same = (not filled) and len(seq_before) == len(seq_after) and np.allclose(seq_before, seq_after, rtol=1e-9)
    if not same:
        st, sk, tt, t_max = _analyze_current(cfg, diag_hint=None)
        timings.update({f"{k}_2": v for k, v in tt.items()})
        t_med = sk.t_median if np.isfinite(sk.t_median) else 1.0
        holes_now = detect_holes(st, sk, t_med)
        holes_kept = [h for h in holes_now if h["is_hole"]]
    # sinon : mêmes faces, mêmes tags -> l'analyse n°1 reste valable telle quelle

    cls = C.classify(sk, cfg)
    kind = cls["kind"]
    mode = reference_skin or cfg["mesh"]["reference_skin"]
    ref, why = C.choose_reference_side(st, sk, kind, mode)
    ref_faces = sorted(f for f, s in sk.side_of_face.items() if s == ref)
    opp_faces = sorted(f for f, s in sk.side_of_face.items() if s != ref)
    ref_tri = np.isin(st.face, ref_faces) & sk.tri_anti & np.isfinite(sk.tri_t)
    zones = C.thickness_zones_weighted(sk.tri_t[ref_tri], st.area[ref_tri],
                                       cfg["analysis"]["constant_thickness_rel_tol"])
    face_curves = {f: sorted(abs(t) for _, t in gmsh.model.getBoundary([(2, f)], oriented=False))
                   for _, f in gmsh.model.getEntities(2)}
    signs = cad_normal_signs(st)
    t0 = time.time()
    for k, ang in exact_dihedrals(st.adjacency, face_curves, signs).items():
        st.adjacency[k] = dict(st.adjacency[k], angle_deg=ang, angle_mesh_deg=st.adjacency[k]["angle_deg"])
    timings["dihedrals"] = time.time() - t0
    bends, sharp = detect_bends(st, sk, ref, cfg)
    refine_bends_cad(bends, st)

    tri_file = workdir / "analysis_tri.npz"
    np.savez_compressed(tri_file, P=st.P, T=st.T, face=st.face, normal=st.normal, size=st.size,
                        area=st.area, centroid=st.centroid)

    min_bend_r = min((b["radius"] for b in bends), default=float("inf"))
    min_hole_d = min((h["diameter"] for h in holes_kept), default=float("inf"))
    fv = feature_vector(kind, cls, diag, dims, len(bends), len(holes_kept), min_bend_r, min_hole_d, len(zones))
    timings["total"] = time.time() - T0
    return PartAnalysis(
        source=str(step_path), brep=str(brep), tri_file=str(tri_file), import_info=dict(
            healing=ir.healing, raw=ir.raw.as_dict(), final=ir.final.as_dict()),
        invariants=inv, exact_hash=ehash, features=fv, diag=diag, obb_dims=[float(x) for x in dims],
        obb_axes=axes.tolist(), kind=kind, classification=cls, reference_side=ref, reference_reason=why,
        ref_faces=ref_faces, opp_faces=opp_faces, flank_faces=sorted(sk.flank_faces),
        failed_faces=st.failed_faces, face_thickness={int(k): float(v) for k, v in sk.face_thickness.items()},
        face_cad_sign=signs, face_curves=face_curves,
        face_adjacency=[[int(a_), int(b_), float(v["angle_deg"])] for (a_, b_), v in st.adjacency.items()],
        thickness_zones=zones, bends=bends,
        sharp_edges=sharp, holes_kept=holes_kept, holes_filled=filled, holes_fill_failed=fill_failed,
        free_edge_length=free_edge_length(st, sk, ref), timings=timings, messages=msgs)
