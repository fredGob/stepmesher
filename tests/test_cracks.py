"""Continuité du maillage : fissures (faces voisines qui ne partagent pas leurs nœuds) et
arêtes non manifold (quality.mesh_cracks, critère dur ; inp_validator)."""
import numpy as np

from stepmesher.io.inp_validator import validate_inp
from stepmesher.mesh.quality import mesh_cracks


def _grid(nx, ny, x0=0.0, y0=0.0, dx=1.0, dy=1.0, off=0):
    X, Y = np.meshgrid(x0 + dx * np.arange(nx + 1), y0 + dy * np.arange(ny + 1), indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    idx = lambda i, j: off + i * (ny + 1) + j  # noqa: E731
    Q = np.array([(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
                  for i in range(nx) for j in range(ny)])
    return P, Q


def _two(left, right):
    (Pa, Qa), (Pb, Qb) = left, right
    return np.r_[Pa, Pb], np.r_[Qa, Qb + len(Pa)]


def test_grille_conforme_sans_fissure():
    P, Q = _grid(6, 4)
    r = mesh_cracks(P, Q)
    assert r["n_nodes"] == 0 and r["n_nonmanifold_edges"] == 0


def test_faces_non_raccordees_fissure():
    """Deux grilles bord à bord (x = 4) sans nœuds communs : 2 x 5 nœuds doublés."""
    P, Q = _two(_grid(4, 4), _grid(4, 4, x0=4.0))
    r = mesh_cracks(P, Q)
    assert r["n_nodes"] == 10


def test_noeuds_pendants_fissure():
    """Découpages différents de part et d'autre (pas 1 contre 0,5) : nœuds pendants."""
    P, Q = _two(_grid(4, 4), _grid(4, 8, x0=4.0, dy=0.5))
    assert mesh_cracks(P, Q)["n_nodes"] >= 5


def test_vraie_fente_non_signalee():
    """Fente réelle de 2 mm entre deux languettes reliées par le fond : pas une fissure."""
    Pa, Qa = _grid(3, 6)                       # languette gauche x 0-3
    Pb, Qb = _grid(3, 6, x0=5.0, off=len(Pa))  # languette droite x 5-8
    Pc, Qc = _grid(8, 2, y0=-2.0, off=len(Pa) + len(Pb))   # fond y -2..0
    P = np.r_[Pa, Pb, Pc]
    Q = np.r_[Qa, Qb, Qc]
    # fusion des nœuds confondus (y = 0) entre languettes et fond
    _, first, inv = np.unique(P.round(9), axis=0, return_index=True, return_inverse=True)
    Q = first[inv.ravel()[Q]]
    assert mesh_cracks(P, Q)["n_nodes"] == 0


def _spike(length):
    """Grille 4 x 4 + bec de 0,002 mm de large et `length` mm de long sous le bord bas (x = 3)."""
    P, Q = _grid(4, 4)
    k = lambda x, y: int(np.nonzero(np.all(np.isclose(P, [x, y, 0]), axis=1))[0][0])  # noqa: E731
    n0 = len(P)
    P = np.r_[P, [[3.002, 0, 0], [3.0, -length, 0], [3.002, -length, 0]]]
    a, b, c, d = k(3, 0), k(4, 0), k(4, 1), k(3, 1)
    Q = np.array([q for q in Q.tolist() if sorted(q) != sorted([a, b, c, d])]
                 + [[n0, b, c, d], [a, n0 + 1, n0 + 2, n0]])
    T = np.array([[a, n0, d]])
    return P, Q, T


def test_replis_microscopiques_ignores():
    """Bec de 0,3 mm x 0,002 mm (arêtes déjà rejetées par le plancher absolu) : ses deux côtés
    se touchent mais sont reliés par le bord en moins de 1 mm -> pas une fissure."""
    P, Q, T = _spike(0.3)
    assert mesh_cracks(P, Q, T)["n_nodes"] == 0
    P, Q, T = _spike(5.0)
    assert mesh_cracks(P, Q, T)["n_nodes"] > 0


def test_arete_non_manifold():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0],
                  [1, 0, 1], [1, 1, 1]], float)
    Q = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [1, 6, 7, 2]])
    assert mesh_cracks(P, Q)["n_nonmanifold_edges"] == 1


def _inp(path, split: bool):
    """Deux SC8R côte à côte ; split : nœuds de l'interface x = 1 doublés."""
    base = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)]
    nodes = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    e1 = [1, 2, 3, 4, 7, 8, 9, 10]
    e2 = [2, 5, 6, 3, 8, 11, 12, 9]
    if split:
        nodes += [(1, 0, 0.0), (1, 1, 0.0), (1, 0, 1.0), (1, 1, 1.0)]
        e2 = [13, 5, 6, 14, 15, 11, 12, 16]
    lines = ["*NODE, NSET=NALL"] + [f"{i + 1}, {x}, {y}, {z}" for i, (x, y, z) in enumerate(nodes)]
    lines += ["*ELEMENT, TYPE=SC8R, ELSET=ES_ALL", "1, " + ", ".join(map(str, e1)), "2, " + ", ".join(map(str, e2)),
              "*SHELL SECTION, ELSET=ES_ALL, MATERIAL=M", "1.0, 5"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_validateur_voit_la_discontinuite(tmp_path):
    ok = validate_inp(_inp(tmp_path / "ok.inp", split=False))
    assert ok["stats"]["crack_nodes"] == 0 and ok["stats"]["coincident_nodes"] == 0
    bad = validate_inp(_inp(tmp_path / "bad.inp", split=True))
    assert not bad["ok"]
    assert bad["stats"]["coincident_nodes"] == 4
    assert bad["stats"]["crack_nodes"] > 0
    assert any("discontinu" in e for e in bad["errors"])
