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
               max_volume_change: float = 5e-3, require_valid: bool = True,
               fillet_max_arc: float = 0.0) -> CleanResult:
    """Nettoie `src` (effondrement des aretes < `precision` mm) et ecrit le resultat
    dans `out_path` s'il est accepte. Accepte seulement si le nombre de solides est
    inchange, le volume varie de moins de `max_volume_change` et (si `require_valid`)
    la forme reste valide. Sinon renvoie la geometrie brute (`src`).

    `fillet_max_arc` > 0 : supprime d'abord les micro-congés de CHANT (face cylindrique
    d'arc <= fillet_max_arc mm dont la hauteur vaut ~l'épaisseur de la tôle) par
    `BRepAlgoAPI_Defeaturing` : les deux chants voisins se prolongent. Leur arc sur la peau
    imposait deux sommets rapprochés -> amas d'éléments minuscules (part_025)."""
    src = str(src)
    try:
        from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.ShapeFix import ShapeFix_Wireframe
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_SOLID
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool
        from OCP.BRepCheck import BRepCheck_Analyzer
    except Exception as e:  # noqa: BLE001 - OCP absent : nettoyage simplement desactive
        return CleanResult("unavailable", src, [f"OCP indisponible ({type(e).__name__})"])
    # `TopTools_IndexedMapOfShape` a disparu de certaines versions d'OCP ; le comptage
    # passe alors par TopExp_Explorer (API stable) avec dé-duplication par hash de shape.
    try:
        from OCP.TopExp import TopExp
        from OCP.TopTools import TopTools_IndexedMapOfShape
    except Exception:  # noqa: BLE001
        TopExp = TopTools_IndexedMapOfShape = None

    if max_tolerance is None:
        max_tolerance = precision

    # Les méthodes statiques d'OCP sont exposées avec ou sans suffixe `_s` selon la
    # version (`TopoDS.Edge_s` vs `TopoDS.Edge`) : on sélectionne celle qui existe.
    def _s(obj, name):
        return getattr(obj, name + "_s") if hasattr(obj, name + "_s") else getattr(obj, name)

    _edge_of = _s(TopoDS, "Edge")
    _degenerated = _s(BRep_Tool, "Degenerated")
    _lin_props = _s(BRepGProp, "LinearProperties")
    _vol_props = _s(BRepGProp, "VolumeProperties")
    _map_shapes = _s(TopExp, "MapShapes") if TopExp is not None else None

    def _iter_kind(shape, kind):
        """Sous-shapes uniques de type `kind` (arête partagée comptée une fois)."""
        if TopTools_IndexedMapOfShape is not None:
            m = TopTools_IndexedMapOfShape()
            _map_shapes(shape, kind, m)
            for i in range(1, m.Extent() + 1):
                yield m.FindKey(i)
            return
        seen = {}
        ex = TopExp_Explorer(shape, kind)
        while ex.More():
            e = ex.Current()
            seen.setdefault(hash(e), e)   # Explorer visite 2x les arêtes partagées
            ex.Next()
        yield from seen.values()

    def _count_kind(shape, kind):
        return sum(1 for _ in _iter_kind(shape, kind))

    def _volume(shape):
        g = GProp_GProps()
        _vol_props(shape, g)
        return g.Mass()

    def _small(shape):
        n = 0
        for s in _iter_kind(shape, TopAbs_EDGE):
            e = _edge_of(s)
            if _degenerated(e):
                continue
            g = GProp_GProps()
            _lin_props(e, g)
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

    n_solids0 = _count_kind(shape, TopAbs_SOLID)
    if n_solids0 == 0:
        return CleanResult("unchanged", src, ["aucun solide : nettoyage OCP ignore (couture gmsh)"])
    messages, fil_stats = [], {}
    if fillet_max_arc > 0:
        shape1, fil_msg, fil_stats = _defeature_chant_fillets(
            shape, fillet_max_arc, max_volume_change, require_valid, n_solids0, _iter_kind,
            _count_kind, _volume)
        if fil_msg:
            messages.append(fil_msg)
        shape = shape1
    n_fil = int(fil_stats.get("removed", 0))
    n_small0 = _small(shape)
    if n_small0 == 0:
        if n_fil:
            return _write(shape, out_path, src, messages, dict(n_solids=n_solids0, n_small=0, **fil_stats))
        return CleanResult("unchanged", src, messages + [f"aucune arete < {precision} mm"],
                           dict(n_solids=n_solids0, n_small=0, **fil_stats))
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

    n_solids1 = _count_kind(cleaned, TopAbs_SOLID)
    n_small1 = _small(cleaned)
    dv = abs(_volume(cleaned) - v0) / max(abs(v0), 1e-30)
    valid = bool(BRepCheck_Analyzer(cleaned).IsValid())
    stats = dict(n_solids=n_solids1, n_small_before=n_small0, n_small_after=n_small1,
                 dv_rel=dv, valid=valid, precision=precision, **fil_stats)

    reject = None
    if n_solids1 != n_solids0:
        reject = f"solides {n_solids0} -> {n_solids1} : nettoyage OCP rejete"
    elif dv > max_volume_change:
        reject = f"dV/V = {dv:.1e} > {max_volume_change:.0e} : nettoyage OCP rejete"
    elif require_valid and not valid:
        reject = "forme invalide apres nettoyage : rejete"
    elif n_small1 >= n_small0:
        reject = f"aucune reduction ({n_small0} -> {n_small1})"
    if reject:
        # micro-arêtes non traitées, mais la suppression des micro-congés reste acquise
        if n_fil:
            return _write(shape, out_path, src, messages + [reject], stats)
        return CleanResult("unchanged" if reject.startswith("aucune") else "rejected", src,
                           messages + [reject], stats)
    return _write(cleaned, out_path, src,
                  messages + [f"micro-aretes < {precision} mm : {n_small0} -> {n_small1}, "
                              f"dV/V = {dv:.1e}, solides = {n_solids1}, valide = {valid}"], stats)


def _write(shape, out_path, src, messages, stats) -> CleanResult:
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    try:
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        w.Write(str(out_path))
    except Exception as e:  # noqa: BLE001
        return CleanResult("error", src, messages + [f"ecriture STEP : {type(e).__name__}"], stats)
    return CleanResult("cleaned", str(out_path), messages, stats)


def _defeature_chant_fillets(shape, max_arc, max_volume_change, require_valid, n_solids0,
                             iter_kind, count_kind, volume):
    """Supprime les micro-congés de chant. Renvoie (forme, message, stats) ; forme
    d'entrée inchangée si rien à faire ou si le résultat est rejeté."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    def _s(obj, name):
        return getattr(obj, name + "_s") if hasattr(obj, name + "_s") else getattr(obj, name)

    face_of, uv_bounds = _s(TopoDS, "Face"), _s(BRepTools, "UVBounds")
    surf_props = _s(BRepGProp, "SurfaceProperties")

    def area(sh):
        g = GProp_GProps()
        surf_props(sh, g)
        return g.Mass()

    v0 = volume(shape)
    t_est = 2.0 * v0 / max(area(shape), 1e-30)    # épaisseur d'une tôle ~ 2V/A
    cands = []
    for sh in iter_kind(shape, TopAbs_FACE):
        f = face_of(sh)
        try:
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() != GeomAbs_Cylinder:
                continue
            umin, umax, vmin, vmax = uv_bounds(f)
            arc = ad.Cylinder().Radius() * abs(umax - umin)
            height = abs(vmax - vmin)
        except Exception:  # noqa: BLE001
            continue
        # chant : hauteur (génératrice) ~ épaisseur ; arc court
        if arc <= max_arc and 0.7 * t_est <= height <= 1.5 * t_est:
            cands.append(f)
    stats = dict(chant_fillets_found=len(cands), removed=0)
    if not cands:
        return shape, "", stats
    try:
        d = BRepAlgoAPI_Defeaturing()
        d.SetShape(shape)
        for f in cands:
            d.AddFaceToRemove(f)
        d.SetRunParallel(False)
        d.Build()
        if not d.IsDone():
            return shape, f"micro-congés de chant : defeaturing en échec ({len(cands)})", stats
        out = d.Shape()
    except Exception as e:  # noqa: BLE001
        return shape, f"micro-congés de chant : {type(e).__name__}", stats
    dv = abs(volume(out) - v0) / max(abs(v0), 1e-30)
    if count_kind(out, TopAbs_SOLID) != n_solids0 or dv > max_volume_change or \
            (require_valid and not BRepCheck_Analyzer(out).IsValid()):
        return shape, f"micro-congés de chant : résultat rejeté (dV/V = {dv:.1e})", stats
    removed = count_kind(shape, TopAbs_FACE) - count_kind(out, TopAbs_FACE)
    stats.update(removed=removed, fillet_dv_rel=dv)
    if removed <= 0:
        return shape, "", stats
    return out, f"micro-congés de chant supprimés : {removed} (arc <= {max_arc} mm, dV/V = {dv:.1e})", stats
