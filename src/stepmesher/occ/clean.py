"""Nettoyage geometrique du STEP via OpenCASCADE (OCP) avant maillage.

Effondre les micro-aretes (slivers d'export CATIA) qui imposeraient des elements
ecrases, tout en preservant un solide valide. Outil : `ShapeFix_Wireframe.FixSmallEdges`,
qui fusionne les petites aretes sur TOUTE la forme de facon coherente (contexte de
partage global) — la ou un healing par contour, un defeaturing ou `UnifySameDomain`
cassent le solide ou n'ont aucun effet sur ces slivers.

Le module est optionnel : si OCP n'est pas installe, `clean_step` renvoie l'etat
`unavailable` et le pipeline continue sur la geometrie brute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CleanResult:
    status: str                        # cleaned | unchanged | rejected | unavailable | error
    path: str                          # chemin STEP a utiliser en aval (nettoye ou source)
    messages: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def clean_step(src, out_path, precision: float = 0.05, max_tolerance: float | None = None,
               drop_small: bool = True, limit_angle: float = -1.0,
               max_volume_change: float = 5e-3, require_valid: bool = True) -> CleanResult:
    """Nettoie `src` (effondrement des aretes < `precision` mm) et ecrit le resultat
    dans `out_path` s'il est accepte. Accepte seulement si le nombre de solides est
    inchange, le volume varie de moins de `max_volume_change` et (si `require_valid`)
    la forme reste valide. Sinon renvoie la geometrie brute (`src`)."""
    src = str(src)
    try:
        from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.ShapeFix import ShapeFix_Wireframe
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_SOLID
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool
        from OCP.BRepCheck import BRepCheck_Analyzer
    except Exception as e:  # noqa: BLE001 - OCP absent : nettoyage simplement desactive
        return CleanResult("unavailable", src, [f"OCP indisponible ({type(e).__name__})"])

    if max_tolerance is None:
        max_tolerance = precision

    def _count(shape, kind):
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, m)
        return m

    def _volume(shape):
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, g)
        return g.Mass()

    def _small(shape):
        m = _count(shape, TopAbs_EDGE)
        n = 0
        for i in range(1, m.Extent() + 1):
            e = TopoDS.Edge_s(m.FindKey(i))
            if BRep_Tool.Degenerated_s(e):
                continue
            g = GProp_GProps()
            BRepGProp.LinearProperties_s(e, g)
            if g.Mass() < precision:
                n += 1
        return n

    try:
        r = STEPControl_Reader()
        if r.ReadFile(src) != IFSelect_RetDone:
            return CleanResult("error", src, ["lecture STEP echouee"])
        r.TransferRoots()
        shape = r.OneShape()
    except Exception as e:  # noqa: BLE001
        return CleanResult("error", src, [f"lecture OCP : {type(e).__name__}"])

    n_solids0 = _count(shape, TopAbs_SOLID).Extent()
    if n_solids0 == 0:
        return CleanResult("unchanged", src, ["aucun solide : nettoyage OCP ignore (couture gmsh)"])
    n_small0 = _small(shape)
    if n_small0 == 0:
        return CleanResult("unchanged", src, [f"aucune arete < {precision} mm"],
                           dict(n_solids=n_solids0, n_small=0))
    v0 = _volume(shape)

    try:
        sfw = ShapeFix_Wireframe(shape)
        sfw.SetPrecision(precision)
        sfw.SetMaxTolerance(max_tolerance)
        sfw.SetLimitAngle(limit_angle)      # -1 : pas de limite d'angle (fusionne non-tangentes)
        sfw.ModeDropSmallEdges = bool(drop_small)
        sfw.FixSmallEdges()
        sfw.FixWireGaps()
        cleaned = sfw.Shape()
    except Exception as e:  # noqa: BLE001
        return CleanResult("error", src, [f"ShapeFix_Wireframe : {type(e).__name__}"],
                           dict(n_solids=n_solids0, n_small=n_small0))

    n_solids1 = _count(cleaned, TopAbs_SOLID).Extent()
    n_small1 = _small(cleaned)
    dv = abs(_volume(cleaned) - v0) / max(abs(v0), 1e-30)
    valid = bool(BRepCheck_Analyzer(cleaned).IsValid())
    stats = dict(n_solids=n_solids1, n_small_before=n_small0, n_small_after=n_small1,
                 dv_rel=dv, valid=valid, precision=precision)

    if n_solids1 != n_solids0:
        return CleanResult("rejected", src,
                           [f"solides {n_solids0} -> {n_solids1} : nettoyage OCP rejete"], stats)
    if dv > max_volume_change:
        return CleanResult("rejected", src,
                           [f"dV/V = {dv:.1e} > {max_volume_change:.0e} : nettoyage OCP rejete"], stats)
    if require_valid and not valid:
        return CleanResult("rejected", src, ["forme invalide apres nettoyage : rejete"], stats)
    if n_small1 >= n_small0:
        return CleanResult("unchanged", src,
                           [f"aucune reduction ({n_small0} -> {n_small1})"], stats)

    try:
        w = STEPControl_Writer()
        w.Transfer(cleaned, STEPControl_AsIs)
        w.Write(str(out_path))
    except Exception as e:  # noqa: BLE001
        return CleanResult("error", src, [f"ecriture STEP : {type(e).__name__}"], stats)
    return CleanResult("cleaned", str(out_path),
                       [f"micro-aretes < {precision} mm : {n_small0} -> {n_small1}, "
                        f"dV/V = {dv:.1e}, solides = {n_solids1}, valide = {valid}"], stats)
