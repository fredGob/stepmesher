"""Contrôle de régularité topologique : étoiles (nœuds intérieurs à 3 ou 5+ quads) dans les
faces qui devraient être maillées en rangées régulières (quality.topology,
quad.face_outline / regular_expected_faces)."""
from types import SimpleNamespace

import gmsh
import numpy as np

from stepmesher.mesh.quad import face_outline, regular_expected_faces
from stepmesher.mesh.quality import topology


def _grid(nx, ny, face, x0=0.0):
    X, Y = np.meshgrid(np.arange(nx + 1) + x0, np.arange(ny + 1), indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    idx = lambda i, j: i * (ny + 1) + j  # noqa: E731
    Q = np.array([(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
                  for i in range(nx) for j in range(ny)])
    inner = np.zeros(len(P), bool)
    for i in range(1, nx):
        for j in range(1, ny):
            inner[idx(i, j)] = True
    return P, Q, np.full(len(Q), face), inner


def _star(face, offset):
    """Triangle découpé en 3 quads : nœud central touché par 3 quads (étoile)."""
    A, B, C = np.array([0, 0, 0.]), np.array([2, 0, 0.]), np.array([1, 1.7, 0.])
    P = np.array([A, B, C, (A + B) / 2, (B + C) / 2, (C + A) / 2, (A + B + C) / 3]) + [offset, 0, 0]
    Q = np.array([(0, 3, 6, 5), (3, 1, 4, 6), (6, 4, 2, 5)])
    inner = np.zeros(7, bool)
    inner[6] = True
    return P, Q, np.full(3, face), inner


def _concat(*parts):
    Ps, Qs, Fs, Is, off = [], [], [], [], 0
    for P, Q, F, inner in parts:
        Ps.append(P)
        Qs.append(Q + off)
        Fs.append(F)
        Is.append(inner)
        off += len(P)
    return np.vstack(Ps), np.vstack(Qs), np.concatenate(Fs), np.concatenate(Is)


def test_grille_reguliere_sans_etoile():
    P, Q, F, inner = _grid(10, 3, face=1)
    t = topology(Q, F, inner, P, {1: {}})
    assert t["irregular_pct"] == 0 and t["n_faces"] == 0 and t["n_irregular"] == 0


def test_etoile_signalee_seulement_dans_une_face_attendue_reguliere():
    X, Q, F, inner = _concat(_star(1, 0.0), _star(2, 10.0), _grid(4, 2, 3, x0=20.0))
    # face 1 attendue régulière (4 coins), face 2 de forme libre : son étoile est normale
    t = topology(Q, F, inner, X, {1: dict(sides=[1, 2, 1, 2]), 3: {}}, min_irregular=1)
    assert t["n_faces"] == 1 and t["faces"][0]["face"] == 1 and t["n_irregular"] == 1
    assert t["faces"][0]["valence_3"] == 1
    assert t["irregular_pct"] > t["pct"] > 0          # l'étoile de la face 2 compte au global seulement


def _plate(points):
    """Surface plane à partir d'un contour de points (polygone fermé)."""
    tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in points]
    lines = [gmsh.model.occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    s = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])
    gmsh.model.occ.synchronize()
    return s


def test_patte_en_biais_decoupee_vue_comme_rectangle():
    """Bande 300 x 10 : grand côté bas découpé en 3, chanfrein de 0,5 mm, bouts en biais
    (profil des pattes de upper part_001) -> 4 coins, attendue régulière. Un L ne l'est pas."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        band = _plate([(0, 0), (100, 0), (180, 0), (300, 0), (306, 10), (6.5, 10), (6, 9.5)])
        o = face_outline(band, short_len=2.0)
        assert o["corners"] == 4 and len(o["sides"]) == 4
        ell = _plate([(400, 0), (500, 0), (500, 10), (420, 10), (420, 60), (400, 60)])
        pa = SimpleNamespace(ref_faces=[band, ell])
        exp = regular_expected_faces(pa, h0=4.0)
        assert band in exp and ell not in exp
    finally:
        gmsh.finalize()


def test_colonnes_redressees():
    """Bande 400 x 20 en grille : bas de 0 à 400, haut décalé (de 40 à 400 : bout gauche en
    biais) -> colonnes penchées ; align_columns les redresse au milieu, bouts figés."""
    from types import SimpleNamespace as NS

    from stepmesher.mesh.repair import align_columns, structured_grid
    n, m = 40, 2
    xs_b = np.linspace(0, 400, n + 1)
    xs_t = np.linspace(40, 400, n + 1)
    rows = [np.c_[xs_b + (xs_t - xs_b) * j / m, np.full(n + 1, 20.0 * j / m), np.zeros(n + 1)] for j in range(m + 1)]
    X = np.vstack(rows)
    idx = lambda i, j: j * (n + 1) + i  # noqa: E731
    Q = np.array([(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)) for j in range(m) for i in range(n)])
    g = structured_grid(Q)
    assert g is not None and g.shape == (m + 1, n + 1)
    qm = NS(X=X.copy(), quads=Q, quad_face=np.full(len(Q), 1))
    before = abs(X[idx(n // 2, m), 0] - X[idx(n // 2, 0), 0])
    r = align_columns(qm, [1], anchors=set())
    assert r["moved"] == 1 and not r["reverted"]
    mid = abs(qm.X[idx(n // 2, m), 0] - qm.X[idx(n // 2, 0), 0])
    assert before > 15 and mid < 1.0
    assert np.all(np.diff(qm.X[[idx(i, m) for i in range(n + 1)], 0]) > 0)   # ordre conservé
