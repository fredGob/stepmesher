"""Bouchage des petits trous débouchants.

Méthode principale (rapide, ~ms par trou) : chaque face de peau percée est
reconstruite sur sa surface support avec ses seules boucles conservées
(`addTrimmedSurface`), puis le solide est recousu sans les parois de trous.
La surface support étant inchangée, le bouchage suit exactement la courbure de
la peau. Contrôle : un seul solide, gain de volume cohérent avec les trous.

Repli : defeaturing OCC (`occ.defeature`), exact mais ~1,5 s par trou.
"""
from __future__ import annotations

import logging
import math

import gmsh

log = logging.getLogger(__name__)


def _volume():
    return sum(gmsh.model.occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))


def fill_by_retrim(holes: list[dict]) -> bool:
    occ = gmsh.model.occ
    wall = sorted({f for h in holes for f in h["faces"]})
    wall_set = set(wall)
    wall_curves = {abs(c) for f in wall for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
    vols = [t for _, t in gmsh.model.getEntities(3)]
    if len(vols) != 1:
        return False
    V0 = _volume()
    expected = sum(math.pi * h["diameter"] ** 2 / 4 * h["thickness"] for h in holes)
    faces = [t for _, t in gmsh.model.getEntities(2) if t not in wall_set]
    new_faces, replaced = [], []
    for f in faces:
        loops, curves = occ.getCurveLoops(f)
        keep = [i for i, cs in enumerate(curves) if not ({abs(int(c)) for c in cs} & wall_curves)]
        if len(keep) == len(loops):
            new_faces.append(f)
            continue
        if not keep:
            return False          # la face n'aurait plus de contour : cas non géré ici
        keep.sort(key=lambda i: -sum(occ.getMass(1, abs(int(c))) for c in curves[i]))
        try:
            # contours existants (gère les arêtes de couture des surfaces périodiques)
            nf = occ.addTrimmedSurface(f, [int(loops[i]) for i in keep], wire3D=True)
        except Exception:  # noqa: BLE001
            nf = occ.addTrimmedSurface(f, [occ.addWire([int(c) for c in curves[i]]) for i in keep], wire3D=True)
        new_faces.append(nf)
        replaced.append(f)
    sl = occ.addSurfaceLoop(new_faces, sewing=True)
    occ.addVolume([sl])
    occ.remove([(3, t) for t in vols])
    occ.remove([(2, t) for t in wall + replaced], recursive=True)
    occ.synchronize()
    n_vol = len(gmsh.model.getEntities(3))
    dV = _volume() - V0
    ok = n_vol == 1 and dV > 0 and abs(dV - expected) <= 0.35 * expected + 1e-9
    if not ok:
        log.warning("bouchage par reconstruction incohérent (volumes=%d, dV=%.4g, attendu %.4g)", n_vol, dV, expected)
    return ok


def fill_holes(holes: list[dict], reimport) -> tuple[list[dict], list[dict], str]:
    """Bouche les trous ; `reimport()` restaure la géométrie de départ en cas d'échec.

    Retourne (bouchés, échecs, méthode).
    """
    if not holes:
        return [], [], "none"
    try:
        if fill_by_retrim(holes):
            return list(holes), [], "retrim"
    except Exception as e:  # noqa: BLE001
        log.warning("bouchage par reconstruction en erreur : %s", e)
    reimport()
    vols = [t for _, t in gmsh.model.getEntities(3)]
    V0 = _volume()
    try:
        gmsh.model.occ.defeature(vols, sorted({f for h in holes for f in h["faces"]}))
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(3)) == len(vols) and _volume() >= V0:
            return list(holes), [], "defeature"
    except Exception as e:  # noqa: BLE001
        log.warning("defeaturing en erreur : %s", e)
    reimport()
    return [], [dict(h, error="bouchage impossible") for h in holes], "failed"
