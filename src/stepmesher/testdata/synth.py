"""Pièces synthétiques de test, générées avec gmsh/OCC, à caractéristiques connues.

Chaque pièce est écrite en STEP ; `EXPECTED` donne ce que l'analyse doit retrouver.
"""
from __future__ import annotations

import math
from pathlib import Path

import gmsh

# épaisseur, nature, nb de plis (congés), trous bouchés / conservés, arêtes vives de la peau intérieure
EXPECTED = {
    "plaque_trouee":  dict(kind="constant", t=2.0, bends=0, holes_filled=2, holes_kept=1),
    "corniere_pliee": dict(kind="constant", t=2.0, bends=1, holes_filled=0, holes_kept=0),
    "profile_u":      dict(kind="constant", t=1.6, bends=2, holes_filled=0, holes_kept=0),
    "bac_angles_vifs": dict(kind="constant", t=1.5, bends=0, holes_filled=0, holes_kept=0),
    "bloc_massif":    dict(kind="massive", t=None, bends=0, holes_filled=0, holes_kept=0),
    "plaque_poches":  dict(kind="variable", t=None, bends=0, holes_filled=0, holes_kept=0),
}


def _fuse_all(tags):
    occ = gmsh.model.occ
    out, _ = occ.fuse([(3, tags[0])], [(3, t) for t in tags[1:]])
    occ.synchronize()
    return out


def _quarter_ring(cx, cy, r, t, a0, L):
    """Quart d'anneau d'axe z, rayons [r, r+t], de l'angle a0 à a0+90°, longueur L."""
    occ = gmsh.model.occ
    c_out = occ.addCylinder(cx, cy, 0, 0, 0, L, r + t, angle=math.pi / 2)
    c_in = occ.addCylinder(cx, cy, 0, 0, 0, L, r, angle=math.pi / 2)
    ring, _ = occ.cut([(3, c_out)], [(3, c_in)])
    occ.rotate(ring, cx, cy, 0, 0, 0, 1, a0)
    return ring[0][1]


def plaque_trouee():
    occ = gmsh.model.occ
    b = occ.addBox(0, 0, 0, 200, 100, 2.0)
    holes = [occ.addCylinder(40, 30, -1, 0, 0, 4, 2.0),     # Ø4 : bouché
             occ.addCylinder(40, 70, -1, 0, 0, 4, 2.0),     # Ø4 : bouché
             occ.addCylinder(130, 50, -1, 0, 0, 4, 15.0)]   # Ø30 : conservé
    occ.cut([(3, b)], [(3, h) for h in holes])


def corniere_pliee():
    occ = gmsh.model.occ
    r, t, L = 4.0, 2.0, 150.0
    ring = _quarter_ring(0, 0, r, t, 0.0, L)                # angle 0..90°
    legA = occ.addBox(r, -60, 0, t, 60, L)                  # prolonge l'extrémité à 0°
    legB = occ.addBox(-40, r, 0, 40, t, L)                  # prolonge l'extrémité à 90°
    _fuse_all([ring, legA, legB])


def profile_u():
    occ = gmsh.model.occ
    r, t, L, W, H = 3.0, 1.6, 200.0, 50.0, 30.0
    # âme horizontale entre deux plis ; ailes verticales vers +y
    ring1 = _quarter_ring(0, 0, r, t, math.pi, L)                   # 180..270°
    ring2 = _quarter_ring(W, 0, r, t, 1.5 * math.pi, L)             # 270..360°
    web = occ.addBox(0, -r - t, 0, W, t, L)
    fl1 = occ.addBox(-r - t, 0, 0, t, H, L)
    fl2 = occ.addBox(W + r, 0, 0, t, H, L)
    _fuse_all([ring1, ring2, web, fl1, fl2])


def bac_angles_vifs():
    occ = gmsh.model.occ
    t = 1.5
    outer = occ.addBox(0, 0, 0, 160, 110, 40)
    inner = occ.addBox(t, t, t, 160 - 2 * t, 110 - 2 * t, 40)
    occ.cut([(3, outer)], [(3, inner)])


def bloc_massif():
    gmsh.model.occ.addBox(0, 0, 0, 50, 40, 30)


def plaque_poches():
    occ = gmsh.model.occ
    b = occ.addBox(0, 0, 0, 200, 100, 6.0)
    p1 = occ.addBox(15, 15, 2.0, 70, 70, 10)
    p2 = occ.addBox(115, 15, 2.0, 70, 70, 10)
    occ.cut([(3, b)], [(3, p1), (3, p2)])


BUILDERS = dict(plaque_trouee=plaque_trouee, corniere_pliee=corniere_pliee, profile_u=profile_u,
                bac_angles_vifs=bac_angles_vifs, bloc_massif=bloc_massif, plaque_poches=plaque_poches)


def write_part(name: str, out: Path, transform: tuple | None = None) -> Path:
    """Écrit la pièce `name` en STEP ; `transform` = (angle, axe(3), translation(3))."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        BUILDERS[name]()
        gmsh.model.occ.removeAllDuplicates()
        if transform:
            ang, ax, tr = transform
            ents = gmsh.model.occ.getEntities(3)
            gmsh.model.occ.rotate(ents, 0, 0, 0, *ax, ang)
            gmsh.model.occ.translate(ents, *tr)
        gmsh.model.occ.synchronize()
        out.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out))
    finally:
        gmsh.finalize()
    return out


def generate_all(outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    return {n: write_part(n, outdir / f"{n}.step") for n in BUILDERS}
