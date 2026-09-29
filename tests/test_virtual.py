"""Topologie virtuelle de la peau (occ/virtual.py) : sommet parasite du contour supprimé dans
le modèle de maillage, arêtes partagées conservées, correspondance des numéros."""
import gmsh
import numpy as np
import pytest

pytest.importorskip("OCP")

from stepmesher.occ.virtual import build_virtual_skin, skin_maps  # noqa: E402


def _two_faces(path):
    """Deux rectangles accolés (arête commune x = 100) ; le bord bas du premier est coupé
    à 1 mm du coin (sommet parasite, raccord tangent) -> 5 courbes au lieu de 4."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    p = [occ.addPoint(*xy, 0) for xy in [(0, 0), (1, 0), (100, 0), (100, 30), (0, 30), (200, 0), (200, 30)]]
    l_a = [occ.addLine(p[0], p[1]), occ.addLine(p[1], p[2]), occ.addLine(p[2], p[3]),
           occ.addLine(p[3], p[4]), occ.addLine(p[4], p[0])]
    l_b = [occ.addLine(p[2], p[5]), occ.addLine(p[5], p[6]), occ.addLine(p[6], p[3])]
    fa = occ.addPlaneSurface([occ.addCurveLoop(l_a)])
    fb = occ.addPlaneSurface([occ.addCurveLoop([l_b[0], l_b[1], l_b[2], l_a[2]])])
    occ.synchronize()
    gmsh.write(str(path))
    info = [dict(tag=f, centre=list(occ.getCenterOfMass(2, f)), area=occ.getMass(2, f)) for f in (fa, fb)]
    old = {}
    for _, c in gmsh.model.getEntities(1):
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        old[c] = np.array(gmsh.model.getValue(1, c, np.linspace(lo[0], hi[0], 7)[1:-1].tolist())).reshape(-1, 3).tolist()
    return info, old, dict(short=l_a[0], long=l_a[1], shared=l_a[2]), fa


def test_sommet_parasite_supprime_arete_partagee_conservee(tmp_path):
    try:
        info, old, cur, fa = _two_faces(tmp_path / "in.brep")
        r = build_virtual_skin(str(tmp_path / "in.brep"), info, str(tmp_path / "skin.brep"), 3.0, 15.0)
        assert r["status"] == "done" and list(r["faces"]) == [fa]
        m = skin_maps(str(tmp_path / "skin.brep"), info, old, tol=0.05)
        cm = m["curve_map"]
        # la courbe de 1 mm et sa voisine tangente ne font plus qu'une ; l'arête commune reste partagée
        assert cm[cur["short"]] == cm[cur["long"]]
        gmsh.model.add("check")
        gmsh.model.occ.importShapes(str(tmp_path / "skin.brep"))
        gmsh.model.occ.synchronize()
        n2o = m["face_map"]
        new_a = next(n for n, o in n2o.items() if o == fa)
        assert len(gmsh.model.getBoundary([(2, new_a)], oriented=False)) == 4
        assert len(gmsh.model.getAdjacencies(1, cm[cur["shared"]])[0]) == 2
    finally:
        gmsh.finalize()


def _gap_face(path, gap):
    """Plaque 40 x 30 dont un coin est un arc R5, prolongé en tangence par un bout de 1 mm dont
    l'extrémité est écartée de `gap` du début de l'arc (tolérance de sommet 1e-5, comme les
    STEP CATIA : echelle part_013, écarts de 5e-6 à 1e-5 mm)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire)
    from OCP.BRepTools import BRepTools
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.gp import gp_Pnt

    P = lambda x, y: gp_Pnt(x, y, 0)  # noqa: E731
    bld = BRep_Builder()

    def V(x, y):
        v = BRepBuilderAPI_MakeVertex(P(x, y)).Vertex()
        bld.UpdateVertex(v, 1e-5)
        return v
    v = [V(0, 0), V(40, 0), V(40, 30), V(6, 30), V(5, 30), V(0, 25)]
    seg = lambda a, b: GC_MakeSegment(P(*a), P(*b)).Value()  # noqa: E731
    arc = GC_MakeArcOfCircle(P(5, 30), P(5 - 5 * np.sqrt(0.5), 25 + 5 * np.sqrt(0.5)), P(0, 25)).Value()
    curves = [(seg((0, 0), (40, 0)), 0, 1), (seg((40, 0), (40, 30)), 1, 2), (seg((40, 30), (6, 30)), 2, 3),
              (seg((6, 30), (5 + gap, 30)), 3, 4), (arc, 4, 5), (seg((0, 25), (0, 0)), 5, 0)]
    mw = BRepBuilderAPI_MakeWire()
    for c, a, b in curves:
        mw.Add(BRepBuilderAPI_MakeEdge(c, v[a], v[b]).Edge())
    face = BRepBuilderAPI_MakeFace(mw.Wire(), True).Face()
    (getattr(BRepTools, "Write_s", None) or BRepTools.Write)(face, str(path))


def test_raccord_tolere_ecart_catia(tmp_path):
    """Bout de 1 mm + arc R5 écartés de 5e-6 mm : fusionnés (avant, tolérance fixe 1e-6 :
    « concaténation impossible », arc à 12 segments de 0,67 mm)."""
    _gap_face(tmp_path / "in.brep", 5e-6)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(tmp_path / "in.brep"))
        gmsh.model.occ.synchronize()
        f = gmsh.model.getEntities(2)[0][1]
        info = [dict(tag=f, centre=list(gmsh.model.occ.getCenterOfMass(2, f)), area=gmsh.model.occ.getMass(2, f))]
        r = build_virtual_skin(str(tmp_path / "in.brep"), info, str(tmp_path / "skin.brep"), 3.0, 15.0)
        assert r["status"] == "done", r["messages"]
        # droite de 34 mm + bout de 1 mm + arc R5 (7,85 mm) : une seule courbe
        assert r["faces"][f] == [pytest.approx(34 + 1 + 2.5 * np.pi, abs=1e-3)]
    finally:
        gmsh.finalize()


def _slices(path):
    """Bande 100 x 20 découpée en x = 40, 41, 43, 44 : 3 tranches (1, 2, 1 mm) entre deux grandes
    faces (soyage de upper part_022, à plat)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    xs = [0, 40, 41, 43, 44, 100]
    rects = [occ.addRectangle(a, 0, 0, b - a, 20) for a, b in zip(xs[:-1], xs[1:])]
    occ.fragment([(2, rects[0])], [(2, r) for r in rects[1:]])
    occ.synchronize()
    gmsh.write(str(path))
    faces = [f for _, f in gmsh.model.getEntities(2)]
    info = [dict(tag=f, centre=list(occ.getCenterOfMass(2, f)), area=occ.getMass(2, f)) for f in faces]
    old = {}
    for _, c in gmsh.model.getEntities(1):
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        old[c] = np.array(gmsh.model.getValue(1, c, np.linspace(lo[0], hi[0], 7)[1:-1].tolist())).reshape(-1, 3).tolist()
    width = {f: occ.getBoundingBox(2, f)[3] - occ.getBoundingBox(2, f)[0] for f in faces}
    return info, old, sorted(f for f in faces if width[f] < 4)


def test_tranches_fusionnees(tmp_path):
    try:
        info, old, slices = _slices(tmp_path / "in.brep")
        assert len(slices) == 3
        r = build_virtual_skin(str(tmp_path / "in.brep"), info, str(tmp_path / "skin.brep"), 3.0, 15.0,
                               max_slice_mm=4.0)
        assert r["status"] == "done", r["messages"]
        assert list(r["groups"].values()) == [slices]
        rep = next(iter(r["groups"]))
        m = skin_maps(str(tmp_path / "skin.brep"), info, old, tol=0.05, groups=r["groups"])
        assert sorted(m["face_map"].values()) == sorted({f["tag"] for f in info} - set(slices[1:]))
        gmsh.model.add("check")
        gmsh.model.occ.importShapes(str(tmp_path / "skin.brep"))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(2)) == 3
        new = next(n for n, o in m["face_map"].items() if o == rep)
        # face fusionnée : 4 côtés, 4 x 20 mm ; bords haut/bas concaténés et partagés par personne
        assert len(gmsh.model.getBoundary([(2, new)], oriented=False)) == 4
        assert gmsh.model.occ.getMass(2, new) == pytest.approx(80.0, rel=1e-6)
        cnt = {}
        for _, f in gmsh.model.getEntities(2):
            for _, c in gmsh.model.getBoundary([(2, f)], oriented=False):
                cnt[abs(c)] = cnt.get(abs(c), 0) + 1
        free = sum(gmsh.model.occ.getMass(1, c) for c, n in cnt.items() if n == 1)
        assert free == pytest.approx(2 * 100 + 2 * 20, rel=1e-9)     # partage intact (pas de fissure)
        assert sum(n == 2 for n in cnt.values()) == 2                  # x = 40 et x = 44
    finally:
        gmsh.finalize()
