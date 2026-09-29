"""Critère de régularité visuelle (quality.regularity) sur des grilles synthétiques."""
from types import SimpleNamespace

import numpy as np

from stepmesher.mesh.quality import regularity


def _grid(xs, ys, face=1):
    """Hexa plats (seule la couche basse compte) sur la grille xs x ys."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    nx, ny = len(xs), len(ys)
    idx = lambda i, j: i * ny + j  # noqa: E731
    q = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
         for i in range(nx - 1) for j in range(ny - 1)]
    hexa = np.array([list(a) + list(a) for a in q])
    return SimpleNamespace(nodes=nodes, hexa=hexa, hexa_face=np.full(len(hexa), face))


def test_grille_uniforme_reguliere():
    sm = _grid(np.arange(0, 101, 10.0), np.arange(0, 51, 10.0))
    r = regularity(sm, 10.0)
    assert r["small_pct"] == 0 and r["jump_pct"] == 0 and r["penalty"] == 0


def test_bande_fine_detectee_et_exemptable():
    xs = np.r_[0, 1, 2, 3, np.arange(13, 104, 10.0)]     # 3 colonnes de 1 mm puis 10 mm
    sm = _grid(xs, np.arange(0, 51, 10.0))
    r = regularity(sm, 10.0)
    assert r["small_pct"] > 20 and r["jump_pct"] > 0
    # la bande fine = pli transfini : exemptée
    sm.hexa_face[: 3 * 5] = 7
    r2 = regularity(sm, 10.0, {7})
    assert r2["small_pct"] == 0 and r2["jump_pct"] == 0
