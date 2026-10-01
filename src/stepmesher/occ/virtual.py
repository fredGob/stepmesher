"""Topologie virtuelle de la peau de référence (modèle de MAILLAGE seulement).

Le contour d'une face de peau est souvent découpé par des sommets sans rôle géométrique :
bout d'une facette de chant (chanfrein de 1,9 mm prolongeant en tangence le bout d'une patte,
upper part_001), arête de 1 mm entre deux chants. Chaque sommet impose un nœud : parité du
full-quad (2 segments de 0,5 mm), rangée de 1,9 mm sur toute la longueur d'une lanière
transfinie, ou quad plat si l'on met un coin transfini dessus. La suppression OCC de la
facette dans le SOLIDE échoue souvent (Defeaturing : 178 s et 0/10 sur part_001).

Ici, sur les seules faces de peau (sans les chants), les courbes consécutives du contour
extérieur reliées par un sommet « parasite » sont concaténées en une BSpline EXACTE (les
courbes d'origine, mises bout à bout) entre les sommets conservés ; la face est reconstruite
sur la même surface ; toutes les autres arêtes sont réutilisées (partage avec les faces
voisines intact). Sommet parasite : n'appartient qu'à cette face de peau (contour libre),
virage de la tangente < max_turn_deg, et une des deux courbes < max_edge_mm.

Le solide et la géométrie d'analyse ne changent pas : correspondance des numéros de faces et
de courbes gmsh établie par `skin_maps` (numérotation du modèle de peau <-> pa.brep).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# écart toléré entre le bout d'une courbe et le début de la suivante lors de la concaténation
JOIN_TOL_MIN_MM = 1e-4


def _s(obj, name):
    return getattr(obj, name + "_s") if hasattr(obj, name + "_s") else getattr(obj, name)


def merge_slices(match: dict, max_slice_mm: float, tangent_deg: float = 5.0, dev_tol_mm: float = 0.05,
                 corner_deg: float = 30.0) -> tuple[dict, dict, list[str]]:
    """Tranches de peau fusionnées (modèle de maillage seulement).

    Tranche = face à une boucle de 4 arêtes dont deux côtés opposés sont courts
    (< max_slice_mm) : sa « largeur » le long de la chaîne. Des tranches consécutives, voisines
    par leurs côtés longs et tangentes le long de ceux-ci (< tangent_deg), forment un groupe
    fusionné en UNE face : surface de Coons sur ses 4 bords, bords courts consécutifs concaténés
    en BSpline exacte (même arête dans les faces voisines, partage conservé). Cas d'origine :
    soyage de 0,9 mm sur 7 mm découpant pli et semelle en tranches de 0,9 à 3,4 mm (upper
    part_022, cadres 000/009) : 6 colonnes de nœuds dans la rampe au lieu de 2.

    match : {tag: TopoDS_Face}. Renvoie (nouveau match, groupes {tag représentant: [tags]},
    messages). Groupe refusé (laissé tel quel) si le contour fusionné n'a pas 4 côtés, si la
    surface s'écarte de plus de dev_tol_mm des tranches ou si la face est invalide."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex,
                                    BRepBuilderAPI_MakeWire)
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
    from OCP.GeomFill import GeomFill_BSplineCurves, GeomFill_CoonsStyle
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.GProp import GProp_GProps
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    face_of, edge_of, wire_of = _s(TopoDS, "Face"), _s(TopoDS, "Edge"), _s(TopoDS, "Wire")
    vertex_of = _s(TopoDS, "Vertex")
    first_v, last_v = _s(TopExp, "FirstVertex"), _s(TopExp, "LastVertex")
    msgs: list[str] = []

    def explore(sh, kind):
        seen = {}
        ex = TopExp_Explorer(sh, kind)
        while ex.More():
            seen.setdefault(hash(ex.Current()), ex.Current())
            ex.Next()
        return list(seen.values())

    def length(e):
        g = GProp_GProps()
        _s(BRepGProp, "LinearProperties")(e, g)
        return g.Mass()

    def area(f):
        g = GProp_GProps()
        _s(BRepGProp, "SurfaceProperties")(f, g)
        return g.Mass()

    def pnt(p):
        return np.array([p.X(), p.Y(), p.Z()])

    def wire_seq(f, wire):
        """[(arête, sens direct)] dans l'ordre de parcours (f : face support, ou None)."""
        seq = []
        wx = BRepTools_WireExplorer(wire, f) if f is not None else BRepTools_WireExplorer(wire)
        while wx.More():
            e = wx.Current()
            seq.append((e, e.Orientation() != TopAbs_REVERSED))
            wx.Next()
        return seq

    why: list[str] = []

    def face_on(surf, outer, inner, ref_area, rtol=2e-3):
        """Face valide sur surf bornée par outer (+ trous), aire à rtol près de ref_area."""
        why.clear()
        for w in (outer, wire_of(outer.Reversed())):
            mf = BRepBuilderAPI_MakeFace(surf, w, True)
            for wi in inner:
                mf.Add(wi)
            if not mf.IsDone():
                why.append(f"MakeFace {mf.Error()}")
                continue
            fix = ShapeFix_Face(mf.Face())
            fix.Perform()
            nf = face_of(fix.Face())
            a = area(nf)
            ok = BRepCheck_Analyzer(nf).IsValid()
            if a > 0 and ok and abs(a - ref_area) <= rtol * ref_area:
                return nf
            why.append(f"aire {a:.2f}/{ref_area:.2f} valide={ok}")
        return None

    def ends(e, fw):
        a, b = first_v(edge_of(e)), last_v(edge_of(e))
        return (a, b) if fw else (b, a)

    def bspline_of(e, forward):
        from OCP.Geom import Geom_TrimmedCurve
        from OCP.GeomConvert import GeomConvert
        ad = BRepAdaptor_Curve(edge_of(e))
        c = ad.Curve().Curve()
        trsf = ad.Trsf()
        c = c.Transformed(trsf) if trsf.Form() != 0 else c
        bs = _s(GeomConvert, "CurveToBSplineCurve")(Geom_TrimmedCurve(c, ad.FirstParameter(), ad.LastParameter()))
        if not forward:
            bs.Reverse()
        return bs

    def normal_at(f, p):
        """Normale sortante (orientation de la face comprise) au point de f le plus proche de p."""
        surf = _s(BRep_Tool, "Surface")(f)
        pr = GeomAPI_ProjectPointOnSurf(p, surf)
        if pr.NbPoints() == 0:
            return None
        u, v = pr.LowerDistanceParameters()
        pr_ = GeomLProp_SLProps(surf, u, v, 1, 1e-9)
        if not pr_.IsNormalDefined():
            return None
        n = pnt(pr_.Normal())
        return -n if f.Orientation() == TopAbs_REVERSED else n

    def turn_at(e1, fw1, e2, fw2):
        """Virage (degrés) entre la fin de e1 et le début de e2 (sens de parcours)."""
        from OCP.gp import gp_Pnt, gp_Vec
        a1, a2 = BRepAdaptor_Curve(edge_of(e1)), BRepAdaptor_Curve(edge_of(e2))
        p, t1, t2 = gp_Pnt(), gp_Vec(), gp_Vec()
        a1.D1(a1.LastParameter() if fw1 else a1.FirstParameter(), p, t1)
        a2.D1(a2.FirstParameter() if fw2 else a2.LastParameter(), p, t2)
        v1, v2 = pnt(t1) * (1 if fw1 else -1), pnt(t2) * (1 if fw2 else -1)
        cs = float(v1 @ v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-300)
        return float(np.degrees(np.arccos(np.clip(cs, -1.0, 1.0))))

    def sample_edge(e, k=3):
        ad = BRepAdaptor_Curve(edge_of(e))
        u0, u1 = ad.FirstParameter(), ad.LastParameter()
        return [ad.Value(u0 + (u1 - u0) * (i + 1) / (k + 1)) for i in range(k)]

    # --- tranches ---
    edge_faces: dict[int, set] = {}
    for tag, f in match.items():
        for e in explore(f, TopAbs_EDGE):
            edge_faces.setdefault(hash(e), set()).add(tag)
    slices = {}
    for tag, f in match.items():
        if len(explore(f, TopAbs_WIRE)) != 1:
            continue
        seq = wire_seq(f, _s(BRepTools, "OuterWire")(f))
        if len(seq) != 4:
            continue
        L = [length(e) for e, _ in seq]
        short = [p for p in ((0, 2), (1, 3)) if max(L[p[0]], L[p[1]]) < max_slice_mm]
        if len(short) != 1:
            continue
        cross = (1, 3) if short[0] == (0, 2) else (0, 2)
        if min(L[cross[0]], L[cross[1]]) <= max(L[short[0][0]], L[short[0][1]]):
            continue
        slices[tag] = dict(cross={hash(seq[i][0]) for i in cross}, edges={hash(seq[i][0]): seq[i][0] for i in cross})
    # --- groupes : tranches voisines par un côté long, tangentes le long de celui-ci ---
    parent = {t: t for t in slices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    cos_t = np.cos(np.radians(tangent_deg))
    for tag, s in slices.items():
        for h in s["cross"]:
            for other in edge_faces.get(h, set()) - {tag}:
                if other not in slices or h not in slices[other]["cross"] or find(other) == find(tag):
                    continue
                ok = True
                for p in sample_edge(s["edges"][h]):
                    na, nb = normal_at(match[tag], p), normal_at(match[other], p)
                    if na is None or nb is None or abs(float(na @ nb)) < cos_t:
                        ok = False
                        break
                if ok:
                    parent[find(other)] = find(tag)
    groups: dict[int, list[int]] = {}
    for t in slices:
        groups.setdefault(find(t), []).append(t)
    groups = {min(m): sorted(m) for m in groups.values() if len(m) >= 2}
    if not groups:
        return match, {}, msgs
    # sommets et arêtes de la peau
    all_edges = {}
    for f in match.values():
        for e in explore(f, TopAbs_EDGE):
            all_edges.setdefault(hash(e), e)
    vert_edges: dict[int, set] = {}
    for h, e in all_edges.items():
        for v in (first_v(edge_of(e)), last_v(edge_of(e))):
            vert_edges.setdefault(hash(v), set()).add(h)

    def attempt(groups):
        """Fusion de tous les groupes avec les sommets libérés par CES groupes. Renvoie
        (faces fusionnées, échecs, messages). Un échec change les sommets libérés : il faut
        recommencer sans lui (sinon arête concaténée d'un côté, pas de l'autre = fissure)."""
        internal = {h for members in groups.values() for h, fs in edge_faces.items()
                    if len(fs & set(members)) >= 2}
        new_edges: dict[frozenset, tuple] = {}

        def removable(v):
            inc = vert_edges.get(hash(v), set())
            return bool(inc & internal) and len(inc - internal) == 2

        def chain_edge(chain):
            """Arête unique d'une chaîne [(arête, sens)] ; la même chaîne vue depuis une autre
            face (parcourue à l'envers) réutilise la même arête."""
            key = frozenset(hash(e) for e, _ in chain)
            if key not in new_edges:
                comp = GeomConvert_CompCurveToBSplineCurve(bspline_of(*chain[0]))
                for e, fw in chain[1:]:
                    tol = max(JOIN_TOL_MIN_MM, 2.0 * _s(BRep_Tool, "Tolerance")(vertex_of(ends(e, fw)[0])))
                    if not comp.Add(bspline_of(e, fw), tol, True):
                        raise RuntimeError("concaténation impossible")
                va, vb = ends(*chain[0])[0], ends(*chain[-1])[1]
                me = BRepBuilderAPI_MakeEdge(comp.BSplineCurve(), vertex_of(va), vertex_of(vb))
                if not me.IsDone():
                    raise RuntimeError("arête concaténée")
                new_edges[key] = (me.Edge(), hash(va))
            return new_edges[key]

        def rebuild(seq):
            """Contour [(arête, sens)] -> fil, chaînes (sommets libérés) remplacées. (fil, nb côtés)."""
            n = len(seq)
            start = next((k for k in range(n) if not removable(ends(*seq[k])[0])), None)
            if start is None:
                raise RuntimeError("contour sans sommet conservé")
            seq = seq[start:] + seq[:start]
            chains, cur = [], [seq[0]]
            for e, fw in seq[1:]:
                if removable(ends(e, fw)[0]):
                    cur.append((e, fw))
                else:
                    chains.append(cur)
                    cur = [(e, fw)]
            chains.append(cur)
            mw = BRepBuilderAPI_MakeWire()
            for chain in chains:
                if len(chain) == 1:
                    e, fw = chain[0]
                    mw.Add(edge_of(e if fw else e.Reversed()))
                    continue
                ne, h_start = chain_edge(chain)
                mw.Add(edge_of(ne if hash(ends(*chain[0])[0]) == h_start else ne.Reversed()))
            if not mw.IsDone():
                raise RuntimeError("contour")
            return mw.Wire(), len(chains)

        merged, failed, msgs_ = {}, [], []
        for rep, members in groups.items():
            try:
                # contour fusionné : arêtes non internes, chaînées par leurs sommets
                bnd = {}
                for t in members:
                    for e in explore(match[t], TopAbs_EDGE):
                        if hash(e) not in internal:
                            bnd[hash(e)] = e
                items = list(bnd.values())
                loop = [(items.pop(0), True)]
                while items:
                    v = hash(ends(*loop[-1])[1])
                    k = next((i for i, e in enumerate(items) if v in (hash(first_v(edge_of(e))), hash(last_v(edge_of(e))))), None)
                    if k is None:
                        raise RuntimeError("contour fusionné ouvert")
                    e = items.pop(k)
                    loop.append((e, hash(first_v(edge_of(e))) == v))
                if hash(ends(*loop[-1])[1]) != hash(ends(*loop[0])[0]):
                    raise RuntimeError("contour fusionné non fermé")
                wire, _ = rebuild(loop)
                # 4 coins réels (plus forts virages) ; un côté peut garder un sommet (tranche
                # voisine non fusionnée) : concaténé pour la seule surface de Coons
                seq = wire_seq(None, wire)
                n = len(seq)
                turns = [turn_at(*seq[i - 1], *seq[i]) for i in range(n)]
                order = sorted(range(n), key=lambda i: -turns[i])
                if n < 4 or turns[order[3]] < corner_deg or (n > 4 and turns[order[4]] >= corner_deg):
                    raise RuntimeError(f"contour fusionné sans 4 coins nets ({n} arêtes)")
                corners = sorted(order[:4])
                curves = []
                for k in range(4):
                    i0, i1 = corners[k], corners[(k + 1) % 4]
                    side = [seq[i % n] for i in range(i0, i1 if i1 > i0 else i1 + n)]
                    comp = GeomConvert_CompCurveToBSplineCurve(bspline_of(*side[0]))
                    for e, fw in side[1:]:
                        if not comp.Add(bspline_of(e, fw), 1e-3, True):
                            raise RuntimeError("côté non concaténable")
                    bs = comp.BSplineCurve()
                    if bs.Degree() < 3:
                        bs.IncreaseDegree(3)     # Coons : au moins 4 pôles par côté (droites de degré 1)
                    bs.SetPole(1, _s(BRep_Tool, "Pnt")(vertex_of(ends(*side[0])[0])))
                    bs.SetPole(bs.NbPoles(), _s(BRep_Tool, "Pnt")(vertex_of(ends(*side[-1])[1])))
                    curves.append(bs)
                surf = GeomFill_BSplineCurves(*curves, GeomFill_CoonsStyle).Surface()
                # aire : contrôle grossier (surface de Coons) ; l'écart à la CAO ci-dessous est le vrai garde-fou
                nf = face_on(surf, wire, [], sum(area(match[t]) for t in members), rtol=0.03)
                if nf is None:
                    raise RuntimeError("face fusionnée invalide : " + " ; ".join(why))
                # écart à la géométrie réelle (nœuds de triangulation des tranches), orientation
                dev, flip = 0.0, 0
                for t in members:
                    BRepMesh_IncrementalMesh(match[t], 0.02, False, 0.3)
                    loc = TopLoc_Location()
                    tri = _s(BRep_Tool, "Triangulation")(match[t], loc)
                    if tri is None:
                        raise RuntimeError("triangulation")
                    tr = loc.Transformation()
                    for i in range(1, tri.NbNodes() + 1):
                        p = tri.Node(i).Transformed(tr)
                        pr = GeomAPI_ProjectPointOnSurf(p, surf)
                        if pr.NbPoints() == 0:
                            raise RuntimeError("projection")
                        dev = max(dev, pr.LowerDistance())
                    n_new, n_old = normal_at(nf, p), normal_at(match[t], p)
                    if n_new is not None and n_old is not None:
                        flip += 1 if float(n_new @ n_old) < 0 else -1
                # ... et dans l'autre sens : la face fusionnée ne déborde pas des tranches
                comp = TopoDS_Compound()
                bld = BRep_Builder()
                bld.MakeCompound(comp)
                for t in members:
                    bld.Add(comp, match[t])
                BRepMesh_IncrementalMesh(nf, 0.02, False, 0.3)
                loc = TopLoc_Location()
                tri = _s(BRep_Tool, "Triangulation")(nf, loc)
                if tri is None:
                    raise RuntimeError("triangulation de la face fusionnée")
                tr = loc.Transformation()
                for i in range(1, tri.NbNodes() + 1):
                    vx = BRepBuilderAPI_MakeVertex(tri.Node(i).Transformed(tr)).Vertex()
                    dd = BRepExtrema_DistShapeShape(vx, comp)
                    if not dd.IsDone():
                        raise RuntimeError("distance")
                    dev = max(dev, dd.Value())
                if dev > dev_tol_mm:
                    raise RuntimeError(f"écart {dev:.3f} mm > {dev_tol_mm}")
                merged[rep] = face_of(nf.Reversed()) if flip > 0 else nf
                msgs_.append(f"tranches {members} fusionnées (écart {dev:.3f} mm)")
            except Exception as e:  # noqa: BLE001
                failed.append(rep)
                msgs_.append(f"tranches {members} : {e}")
        if failed:
            return None, failed, msgs_
        # faces voisines : chaînes remplacées par la même arête concaténée
        grouped = {t for m in groups.values() for t in m}
        out = {t: f for t, f in match.items() if t not in grouped}
        for tag, f in list(out.items()):
            if not any(removable(v) for e in explore(f, TopAbs_EDGE) for v in (first_v(edge_of(e)), last_v(edge_of(e)))):
                continue
            ff = face_of(f.Oriented(TopAbs_FORWARD))
            outer = _s(BRepTools, "OuterWire")(ff)
            w_out, _ = rebuild(wire_seq(ff, outer))
            inner = [rebuild(wire_seq(ff, wire_of(w)))[0] for w in explore(ff, TopAbs_WIRE) if not w.IsSame(outer)]
            nf = face_on(_s(BRep_Tool, "Surface")(ff), w_out, inner, area(ff))
            if nf is None:
                raise RuntimeError(f"face voisine {tag} non reconstruite")
            out[tag] = face_of(nf.Reversed()) if f.Orientation() == TopAbs_REVERSED else nf
        out.update(merged)
        return out, [], msgs_

    for _ in range(5):
        try:
            out, failed, m = attempt(groups)
        except Exception as e:  # noqa: BLE001
            msgs.append(f"fusion des tranches annulée : {e}")
            return match, {}, msgs
        msgs += m
        if not failed:
            return out, groups, msgs
        groups = {r: g for r, g in groups.items() if r not in failed}
        if not groups:
            break
    return match, {}, msgs


def build_virtual_skin(brep_in: str, ref_faces: list[dict], out_path: str, max_edge_mm: float = 3.0,
                       max_turn_deg: float = 15.0, max_slice_mm: float = 0.0, slice_dev_mm: float = 0.05) -> dict:
    """ref_faces : [dict(tag, centre, area)] des faces de peau (numérotation gmsh de brep_in).
    Écrit un BREP (compound des seules faces de peau, contours simplifiés) dans out_path.
    max_slice_mm > 0 : tranches étroites tangentes fusionnées d'abord (merge_slices).
    Renvoie dict(status = done | unchanged | error, faces = {tag: courbes fusionnées},
    groups = {tag représentant: [tags des tranches fusionnées]}, messages)."""
    try:
        from OCP.BRep import BRep_Builder, BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
        from OCP.Geom import Geom_TrimmedCurve
        from OCP.GeomConvert import GeomConvert, GeomConvert_CompCurveToBSplineCurve
        from OCP.GProp import GProp_GProps
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.ShapeFix import ShapeFix_Face
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_VERTEX, TopAbs_WIRE
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
    except Exception as e:  # noqa: BLE001
        return dict(status="error", faces={}, messages=[f"OCP indisponible ({type(e).__name__})"])

    face_of, edge_of, wire_of = _s(TopoDS, "Face"), _s(TopoDS, "Edge"), _s(TopoDS, "Wire")
    vertex_of = _s(TopoDS, "Vertex")
    surf_props, lin_props = _s(BRepGProp, "SurfaceProperties"), _s(BRepGProp, "LinearProperties")

    def explore(sh, kind):
        seen = {}
        ex = TopExp_Explorer(sh, kind)
        while ex.More():
            seen.setdefault(hash(ex.Current()), ex.Current())
            ex.Next()
        return list(seen.values())

    def area_centre(f):
        g = GProp_GProps()
        surf_props(f, g)
        c = g.CentreOfMass()
        return g.Mass(), np.array([c.X(), c.Y(), c.Z()])

    def length(e):
        g = GProp_GProps()
        lin_props(e, g)
        return g.Mass()

    shape = TopoDS_Shape()
    try:
        _s(BRepTools, "Read")(shape, str(brep_in), BRep_Builder())
    except Exception as e:  # noqa: BLE001
        return dict(status="error", faces={}, messages=[f"lecture BREP : {type(e).__name__}"])
    occ_faces = [face_of(f) for f in explore(shape, TopAbs_FACE)]
    props = [area_centre(f) for f in occ_faces]
    # correspondance faces gmsh <-> faces OCP (aire et centre de gravité)
    match = {}
    for rf in ref_faces:
        c = np.asarray(rf["centre"], float)
        best = min(range(len(occ_faces)), key=lambda i: np.linalg.norm(props[i][1] - c)
                   + abs(props[i][0] - rf["area"]) / max(rf["area"], 1e-12))
        if abs(props[best][0] - rf["area"]) > 1e-6 * max(rf["area"], 1.0) + 1e-9 or \
                np.linalg.norm(props[best][1] - c) > 1e-4 * (1.0 + np.linalg.norm(c)):
            return dict(status="error", faces={}, messages=[f"face {rf['tag']} introuvable dans le BREP"])
        match[int(rf["tag"])] = occ_faces[best]
    # arêtes et sommets -> faces de peau qui les utilisent
    edge_faces: dict[int, set] = {}
    vert_faces: dict[int, set] = {}
    for tag, f in match.items():
        for e in explore(f, TopAbs_EDGE):
            edge_faces.setdefault(hash(e), set()).add(tag)
        for v in explore(f, TopAbs_VERTEX):
            vert_faces.setdefault(hash(v), set()).add(tag)

    def traversal(e, forward):
        """Tangentes unitaires (départ, arrivée) dans le sens de parcours."""
        ad = BRepAdaptor_Curve(e)
        u0, u1 = ad.FirstParameter(), ad.LastParameter()
        out = []
        for u in (u0, u1):
            p, v = gp_Pnt(), gp_Vec()
            ad.D1(u, p, v)
            out.append(np.array([v.X(), v.Y(), v.Z()]))
        t0, t1 = out if forward else (-out[1], -out[0])
        return t0 / max(np.linalg.norm(t0), 1e-300), t1 / max(np.linalg.norm(t1), 1e-300)

    def bspline_of(e, forward):
        ad = BRepAdaptor_Curve(e)
        c = ad.Curve().Curve()
        trsf = ad.Trsf()
        c = c.Transformed(trsf) if trsf.Form() != 0 else c
        bs = _s(GeomConvert, "CurveToBSplineCurve")(Geom_TrimmedCurve(c, ad.FirstParameter(), ad.LastParameter()))
        if not forward:
            bs.Reverse()
        return bs

    new_faces, report, msgs = {}, {}, []
    for tag, face in match.items():
        ff = face_of(face.Oriented(TopAbs_FORWARD))
        outer = _s(BRepTools, "OuterWire")(ff)
        seq = []                                  # (arête, sens direct, sommet de départ)
        wx = BRepTools_WireExplorer(outer, ff)
        while wx.More():
            e = wx.Current()
            seq.append((e, e.Orientation() != TopAbs_REVERSED, wx.CurrentVertex()))
            wx.Next()
        n = len(seq)
        if n < 3:
            continue
        L = [length(e) for e, _, _ in seq]
        T = [traversal(e, fw) for e, fw, _ in seq]
        soft = []                                  # soft[k] : sommet entre seq[k] et seq[k+1]
        for k in range(n):
            j = (k + 1) % n
            v = seq[j][2]
            turn = float(np.degrees(np.arccos(np.clip(T[k][1] @ T[j][0], -1.0, 1.0))))
            soft.append(vert_faces.get(hash(v), set()) == {tag}
                        and edge_faces.get(hash(seq[k][0]), set()) == {tag}
                        and edge_faces.get(hash(seq[j][0]), set()) == {tag}
                        and turn < max_turn_deg and min(L[k], L[j]) < max_edge_mm)
        if not any(soft) or all(soft):
            continue
        # chaînes d'arêtes reliées par des sommets parasites (parcours circulaire depuis un sommet dur)
        start = next(k for k in range(n) if not soft[k])
        runs, cur = [], []
        for d in range(1, n + 1):
            k = (start + d) % n
            cur.append(k)
            if not soft[k]:
                runs.append(cur)
                cur = []
        try:
            mw = BRepBuilderAPI_MakeWire()
            merged = []
            for run in runs:
                if len(run) == 1:
                    e, fw, _ = seq[run[0]]
                    mw.Add(edge_of(e))
                    continue
                comp = GeomConvert_CompCurveToBSplineCurve(bspline_of(seq[run[0]][0], seq[run[0]][1]))
                for k in run[1:]:
                    # bouts de courbes CATIA écartés de ~1e-5 mm (arc R5 + bouts de 0,86 mm,
                    # echelle part_013) : raccord à la tolérance du sommet, 1e-6 refusait tout
                    tol_v = max(JOIN_TOL_MIN_MM, 2.0 * _s(BRep_Tool, "Tolerance")(vertex_of(seq[k][2])))
                    if not comp.Add(bspline_of(seq[k][0], seq[k][1]), tol_v, True):
                        raise RuntimeError("concaténation impossible")
                v_a = vertex_of(seq[run[0]][2])
                v_b = vertex_of(seq[(run[-1] + 1) % n][2])
                me = BRepBuilderAPI_MakeEdge(comp.BSplineCurve(), v_a, v_b)
                if not me.IsDone():
                    raise RuntimeError("arête fusionnée")
                mw.Add(me.Edge())
                merged.append(round(sum(L[k] for k in run), 3))
            if not mw.IsDone():
                raise RuntimeError("contour")
            a_old, _ = area_centre(ff)
            nf, a_new = None, 0.0
            # sens du nouveau contour : celui qui redonne une aire positive
            for wire in (mw.Wire(), wire_of(mw.Wire().Reversed())):
                mf = BRepBuilderAPI_MakeFace(_s(BRep_Tool, "Surface")(ff), wire, True)
                for w in explore(ff, TopAbs_WIRE):
                    if not w.IsSame(outer):
                        mf.Add(wire_of(w))
                if not mf.IsDone():
                    continue
                fix = ShapeFix_Face(mf.Face())
                fix.Perform()
                nf = fix.Face()
                a_new, _ = area_centre(nf)
                if a_new > 0:
                    break
            if nf is None:
                raise RuntimeError("face")
            if not BRepCheck_Analyzer(nf).IsValid() or abs(a_new - a_old) > 1e-3 * a_old:
                raise RuntimeError(f"face invalide (aire {a_old:.1f} -> {a_new:.1f})")
            if face.Orientation() == TopAbs_REVERSED:
                nf = face_of(nf.Reversed())
            new_faces[tag] = nf
            report[tag] = merged
        except Exception as e:  # noqa: BLE001
            msgs.append(f"face {tag} : {e}")
    # tranches étroites tangentes fusionnées APRÈS la simplification des contours (un sommet
    # parasite sur le bord libre d'une tranche lui donnait 5 arêtes : tranche non reconnue)
    match = {tag: new_faces.get(tag, f) for tag, f in match.items()}
    groups = {}
    if max_slice_mm > 0:
        try:
            match, groups, m = merge_slices(match, max_slice_mm, dev_tol_mm=slice_dev_mm)
            msgs += m
        except Exception as e:  # noqa: BLE001
            groups = {}
            msgs.append(f"fusion des tranches : {type(e).__name__}: {e}")
    if not new_faces and not groups:
        return dict(status="unchanged", faces={}, groups={}, messages=msgs)
    comp_ = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp_)
    for tag, f in match.items():
        b.Add(comp_, f)
    _s(BRepTools, "Write")(comp_, str(out_path))
    return dict(status="done", faces={int(k): v for k, v in report.items()},
                groups={int(k): [int(t) for t in v] for k, v in groups.items()}, messages=msgs)


def skin_maps(skin_brep: str, ref_faces: list[dict], old_curves: dict, tol: float, groups: dict | None = None) -> dict:
    """Correspondances entre le modèle de peau virtuel (importé dans un modèle gmsh temporaire)
    et la numérotation de pa.brep : faces (neuves -> anciennes, par aire + centre) et courbes
    (anciennes -> neuves, par points échantillonnés sur l'ancienne courbe). old_curves :
    {tag ancien: [points (x, y, z)]}. groups : {représentant: [faces fusionnées]} -> la face
    neuve correspond au représentant (aire et centre du groupe). Le modèle gmsh courant est restauré."""
    groups = groups or {}
    by_tag = {int(rf["tag"]): rf for rf in ref_faces}
    member = {t for g in groups.values() for t in g}
    eff = [rf for rf in ref_faces if int(rf["tag"]) not in member]
    for rep, g in groups.items():
        A = sum(by_tag[t]["area"] for t in g)
        C = sum(np.asarray(by_tag[t]["centre"], float) * by_tag[t]["area"] for t in g) / max(A, 1e-300)
        eff.append(dict(tag=int(rep), centre=C.tolist(), area=A))
    ref_faces = eff
    import gmsh
    from scipy.spatial import cKDTree
    cur = gmsh.model.getCurrent()
    gmsh.model.add("_virtual_skin")
    try:
        gmsh.model.occ.importShapes(str(skin_brep))
        gmsh.model.occ.synchronize()
        new = {}
        for _, f in gmsh.model.getEntities(2):
            new[f] = (gmsh.model.occ.getMass(2, f), np.array(gmsh.model.occ.getCenterOfMass(2, f)))
        face_map = {}
        for rf in ref_faces:
            c = np.asarray(rf["centre"], float)
            f = min(new, key=lambda k: np.linalg.norm(new[k][1] - c) + abs(new[k][0] - rf["area"]) / max(rf["area"], 1e-12))
            face_map[int(f)] = int(rf["tag"])
        if len(set(face_map.values())) != len(ref_faces) or len(face_map) != len(new):
            raise RuntimeError("faces non appariées")
        # polylignes denses des courbes neuves ; distance point-segment (les points des
        # anciennes courbes sont SUR une courbe neuve : identique ou concaténée)
        seg_a, seg_b, owner = [], [], []
        for _, c in gmsh.model.getEntities(1):
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            P = np.array(gmsh.model.getValue(1, c, np.linspace(lo[0], hi[0], 401).tolist())).reshape(-1, 3)
            seg_a.append(P[:-1])
            seg_b.append(P[1:])
            owner += [c] * (len(P) - 1)
        A, B, owner = np.vstack(seg_a), np.vstack(seg_b), np.array(owner)
        tree = cKDTree(0.5 * (A + B))
        curve_map = {}
        for oc, samp in old_curves.items():
            S = np.asarray(samp, float)
            _, J = tree.query(S, k=min(8, len(A)))
            J = np.atleast_2d(J)
            best_c, ok = [], True
            for s_, js in zip(S, J):
                a, b = A[js], B[js]
                ab = b - a
                t = np.clip(np.einsum("ij,ij->i", s_ - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
                d = np.linalg.norm(a + t[:, None] * ab - s_, axis=1)
                k = int(np.argmin(d))
                if d[k] > tol:
                    ok = False
                    break
                best_c.append(owner[js[k]])
            if ok and best_c:
                vals, cnt = np.unique(best_c, return_counts=True)
                curve_map[int(oc)] = int(vals[np.argmax(cnt)])
        return dict(face_map=face_map, curve_map=curve_map)
    finally:
        gmsh.model.remove()
        if cur:
            gmsh.model.setCurrent(cur)
