"""Import STEP via gmsh/OpenCASCADE et nettoyage non destructif."""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

import gmsh
import numpy as np

log = logging.getLogger(__name__)


@contextlib.contextmanager
def gmsh_session(cfg=None, verbose: bool = False):
    """Session gmsh isolée. Toujours finalisée, même en cas d'exception."""
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 1)
        if cfg is not None:
            gmsh.option.setString("Geometry.OCCTargetUnit", cfg["general"]["occ_target_unit"])
            nt = int(cfg["general"]["threads"])
            gmsh.option.setNumber("General.NumThreads", nt)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", nt)
        gmsh.model.add("part")
        yield
    finally:
        gmsh.finalize()


@dataclass
class GeomState:
    n_solids: int
    n_faces: int
    n_shells_free: int
    volume: float
    area: float

    def as_dict(self):
        return dict(n_solids=self.n_solids, n_faces=self.n_faces, volume=self.volume, area=self.area)


def geom_state() -> GeomState:
    vols = gmsh.model.getEntities(3)
    faces = gmsh.model.getEntities(2)
    V = float(sum(gmsh.model.occ.getMass(3, t) for _, t in vols))
    A = float(sum(gmsh.model.occ.getMass(2, t) for _, t in faces))
    return GeomState(len(vols), len(faces), 0, V, A)


@dataclass
class ImportResult:
    raw: GeomState
    final: GeomState
    healing: str                      # "none" | "applied" | "rejected" | "sewn"
    messages: list[str] = field(default_factory=list)


def _clear():
    gmsh.model.occ.remove(gmsh.model.occ.getEntities(), recursive=True)
    gmsh.model.occ.synchronize()


def _raw_import(path: str, small_edge_tol: float | None = None):
    """Import OCC ; avec `small_edge_tol`, suppression des micro-arêtes à l'import
    (ShapeFix OCC). Les options sont toujours remises à zéro ensuite, pour que les
    relectures de la géométrie préparée (.brep) ne la modifient plus."""
    _clear()
    if small_edge_tol:
        gmsh.option.setNumber("Geometry.Tolerance", small_edge_tol)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
    try:
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
    finally:
        gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)


def _clear_and_import(path: str):
    _raw_import(path)


def micro_curve_count(max_len: float) -> int:
    return sum(1 for _, c in gmsh.model.getEntities(1) if gmsh.model.occ.getMass(1, c) < max_len)


def import_step(path: str, cfg) -> ImportResult:
    """Importe le STEP et applique un nettoyage qui ne dégrade jamais un solide valide.

    1. micro-arêtes (exports CATIA : arêtes de 0,001 à 0,02 mm qui imposent des
       éléments écrasés) : réimport avec correction OCC, tolérances croissantes ;
       accepté seulement si le nombre de solides est inchangé et le volume varie
       de moins de `max_volume_change` ;
    2. fichier sans solide : couture + reconstruction de solide ;
    3. faces dégénérées (+ micro-faces en mode agressif), avec le même contrôle.
    """
    msgs: list[str] = []
    h = cfg["healing"]
    _raw_import(path)
    raw = geom_state()
    method = "none"

    tols = [float(x) for x in h.get("small_edge_tol_mm", [])] if h["enabled"] else []
    if tols and raw.n_solids > 0:
        n0 = micro_curve_count(max(tols))
        if n0:
            accepted = False
            for tol in tols:
                try:
                    _raw_import(path, tol)
                    st = geom_state()
                except Exception as e:  # noqa: BLE001
                    msgs.append(f"micro-arêtes, tolérance {tol} mm : échec OCC ({str(e)[:60]})")
                    continue
                dv = abs(st.volume - raw.volume) / max(abs(raw.volume), 1e-30)
                n1 = micro_curve_count(max(tols))
                if st.n_solids == raw.n_solids and dv <= h["max_volume_change"] and n1 < n0:
                    msgs.append(f"micro-arêtes supprimées (tolérance {tol} mm) : {n0} -> {n1} courbes "
                                f"< {max(tols)} mm, dV/V = {dv:.1e}")
                    method = f"micro_edges({tol})"
                    accepted = True
                    break
                msgs.append(f"micro-arêtes, tolérance {tol} mm : rejeté (solides {st.n_solids}, dV/V={dv:.1e})")
            if not accepted:
                _raw_import(path)
                msgs.append(f"{n0} micro-arête(s) conservée(s)")

    if raw.n_solids == 0:
        msgs.append("aucun solide dans le fichier : couture et reconstruction de solide")
        gmsh.model.occ.healShapes(gmsh.model.occ.getEntities(2), tolerance=max(h["tolerance"], 1e-3),
                                  fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
                                  sewFaces=True, makeSolids=True)
        gmsh.model.occ.synchronize()
        st = geom_state()
        if st.n_solids == 0:
            msgs.append("reconstruction de solide impossible")
        return ImportResult(raw, st, "sewn", msgs)

    if not h["enabled"]:
        return ImportResult(raw, raw, "none", msgs)

    before = geom_state()
    try:
        ag = bool(h["aggressive"])
        gmsh.model.occ.healShapes(gmsh.model.occ.getEntities(3), tolerance=h["tolerance"],
                                  fixDegenerated=True, fixSmallEdges=ag, fixSmallFaces=ag,
                                  sewFaces=False, makeSolids=False)
        gmsh.model.occ.synchronize()
        st = geom_state()
        dv = abs(st.volume - before.volume) / max(abs(before.volume), 1e-30)
        if st.n_solids != before.n_solids or st.volume <= 0 or dv > h["volume_rel_tol"]:
            raise RuntimeError(f"solides {before.n_solids}->{st.n_solids}, dV/V={dv:.2e}")
        if st.n_faces != before.n_faces:
            msgs.append(f"nettoyage : {before.n_faces} -> {st.n_faces} faces")
        return ImportResult(raw, st, method + "+applied" if method != "none" else "applied", msgs)
    except Exception as e:  # noqa: BLE001 - le nettoyage ne doit jamais faire échouer la pièce
        msgs.append(f"nettoyage des faces dégénérées rejeté ({e})")
        tol = float(method[12:-1]) if method.startswith("micro_edges") else None
        _raw_import(path, tol)
        return ImportResult(raw, geom_state(), method if method != "none" else "rejected", msgs)


def aabb_diag() -> float:
    bb = gmsh.model.getBoundingBox(-1, -1)
    return float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
