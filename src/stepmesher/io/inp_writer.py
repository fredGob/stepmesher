"""Export Abaqus .inp : SC8R (+ SC6R), sets, surfaces, orientation par élément, sections.

Aucun bloc matériau : les sections référencent MATERIAL=TBD, à définir par l'utilisateur.

Note continuum shell : l'épaisseur mécanique d'un SC8R est portée par la géométrie
nodale ; la valeur écrite dans *SHELL SECTION est l'épaisseur nominale mesurée de la
zone (à confirmer lors du datacheck Abaqus, cf. README).
"""
from __future__ import annotations

import datetime as dt
import io
import unicodedata
from pathlib import Path

import numpy as np

from .. import __version__


def _ids(f, ids, per_line=16):
    ids = list(ids)
    for k in range(0, len(ids), per_line):
        f.write(", ".join(str(int(i)) for i in ids[k:k + per_line]) + "\n")


def to_ascii(text: str) -> str:
    """Abaqus peut refuser les caractères non ASCII, même en commentaire."""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("°", " deg").replace("×", "x")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def assign_zones(thickness: np.ndarray, zones: list[float]) -> tuple[np.ndarray, list[float]]:
    """Zone d'épaisseur de chaque élément (palier le plus proche, en échelle log)."""
    if not zones:
        zones = [float(np.median(thickness))] if len(thickness) else [1.0]
    z = np.log(np.asarray(zones))
    k = np.argmin(np.abs(np.log(np.maximum(thickness, 1e-12))[:, None] - z[None, :]), axis=1)
    return k, list(zones)


def skin_areas(X: np.ndarray, H: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Aires des faces S1 (nœuds 1-4 / 1-3) et S2 (5-8 / 4-6) sur l'ensemble des éléments."""
    def quad(c):
        if not len(c):
            return 0.0
        a, b, cc, d = (X[c[:, i]] for i in range(4))
        return float(0.5 * np.linalg.norm(np.cross(cc - a, d - b), axis=1).sum())

    def tri(c):
        if not len(c):
            return 0.0
        a, b, cc = (X[c[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum())
    H = np.asarray(H, int).reshape(-1, 8)
    W = np.asarray(W, int).reshape(-1, 6)
    return quad(H[:, :4]) + tri(W[:, :3]), quad(H[:, 4:]) + tri(W[:, 3:])


def write_inp(path: Path, mesh: dict, pa, recipe: dict, metrics: dict, status: str) -> dict:
    nodes = mesh["nodes"]
    N = int(mesh["n_ref"])
    H, W = mesh["hexa"], mesh["wedge"]
    nh, nw = len(H), len(W)
    n_el = nh + nw
    faces = np.r_[mesh["hexa_face"], mesh["wedge_face"]].astype(int)
    thick = mesh["thickness"]
    zone_of, zones = assign_zones(thick, pa.thickness_zones if pa.kind == "variable" else
                                  [pa.classification["t_median"]])
    # zones effectivement utilisées, épaisseur de section = moyenne des éléments de la zone
    used = sorted(set(zone_of.tolist()))
    zone_t = {z: float(np.mean(thick[zone_of == z])) for z in used}
    bend_faces = {f for b in pa.bends for f in b["faces"]}
    is_bend = np.isin(faces, list(bend_faces))
    soft = mesh.get("soft")
    eid = np.arange(1, n_el + 1)

    f = io.StringIO()
    if True:
        f.write("*HEADING\n")
        f.write(f"stepmesher {__version__} - {Path(pa.source).name} - {status}\n")
        f.write(f"** généré le {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"** source : {pa.source}\n")
        f.write(f"** nature : {pa.kind} ; épaisseur médiane {pa.classification['t_median']:.4g} mm ; "
                f"peau de référence : {pa.reference_reason}\n")
        f.write(f"** recette : {recipe}\n")
        f.write(f"** éléments : {nh} SC8R, {nw} SC6R ; 1 élément dans l'épaisseur\n")
        f.write("** nœuds 1-4 (1-3) sur la peau de référence, 5-8 (4-6) sur la peau opposée\n")
        f.write("** MATERIAL=TBD : bloc matériau à fournir\n")
        f.write("*NODE, NSET=NALL\n")
        for i, p in enumerate(nodes, 1):
            f.write(f"{i}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}\n")
        if nh:
            f.write("*ELEMENT, TYPE=SC8R, ELSET=ES_SC8R\n")
            for k, c in enumerate(H, 1):
                f.write(f"{k}, " + ", ".join(str(int(x) + 1) for x in c) + "\n")
        if nw:
            f.write("*ELEMENT, TYPE=SC6R, ELSET=ES_SC6R\n")
            for k, c in enumerate(W, nh + 1):
                f.write(f"{k}, " + ", ".join(str(int(x) + 1) for x in c) + "\n")

        # --- node sets ---
        f.write("*NSET, NSET=NS_SKIN_REF, GENERATE\n")
        f.write(f"1, {N}, 1\n")
        f.write("*NSET, NSET=NS_SKIN_OPP, GENERATE\n")
        f.write(f"{N + 1}, {2 * N}, 1\n")
        nsets = {}
        for name, key in (("NS_FREE_EDGES", "ns_free"), ("NS_HOLE_EDGES", "ns_hole")):
            ids = np.asarray(mesh.get(key, np.zeros(0, int)), int)
            if len(ids):
                f.write(f"*NSET, NSET={name}\n")
                _ids(f, np.r_[ids + 1, ids + 1 + N])
                nsets[name] = int(2 * len(ids))

        # --- element sets ---
        f.write("*ELSET, ELSET=ES_ALL\n")
        f.write(", ".join(s for s, n in (("ES_SC8R", nh), ("ES_SC6R", nw)) if n) + "\n")
        elsets = {}
        for name, mask in (("ES_BENDS", is_bend), ("ES_FLAT", ~is_bend)):
            if mask.any():
                f.write(f"*ELSET, ELSET={name}\n")
                _ids(f, eid[mask])
                elsets[name] = int(mask.sum())
        zone_names = {}
        for z in used:
            name = f"ES_ZONE_{len(zone_names) + 1}"
            zone_names[z] = name
            f.write(f"** zone {name} : épaisseur ~{zone_t[z]:.4g} mm\n")
            f.write(f"*ELSET, ELSET={name}\n")
            _ids(f, eid[zone_of == z])
            elsets[name] = int((zone_of == z).sum())
        if soft is not None and np.any(soft):
            f.write("** éléments hors critères cibles (acceptés dans la tolérance configurée)\n")
            f.write("*ELSET, ELSET=ES_QUALITY_WARN\n")
            _ids(f, eid[np.asarray(soft, bool)])
            elsets["ES_QUALITY_WARN"] = int(np.sum(soft))

        # --- surfaces des deux peaux externes ---
        # S1 = face nœuds 1-4 (peau de référence), S2 = face 5-8 (peau opposée).
        # Intérieure / extérieure : la peau d'aire la plus faible est l'intérieure (côté
        # concave des plis) ; sur une plaque sans pli les deux aires sont quasi égales.
        a_ref, a_opp = skin_areas(nodes, H, W)
        inner_face = "S1" if a_ref <= a_opp else "S2"
        outer_face = "S2" if inner_face == "S1" else "S1"
        rel = abs(a_ref - a_opp) / max(a_ref, a_opp, 1e-30)
        f.write(f"** aires des peaux : S1 (référence) {a_ref:.6g} mm2, S2 (opposée) {a_opp:.6g} mm2"
                f" (écart {100 * rel:.2f} %)\n")
        if rel < 0.005:
            f.write("** écart < 0,5 % : pièce quasi plane, distinction intérieure/extérieure peu significative\n")
        surfaces = {}
        for name, face in (("SURF_REF", "S1"), ("SURF_OPP", "S2"),
                           ("SURF_INNER", inner_face), ("SURF_OUTER", outer_face)):
            f.write(f"*SURFACE, TYPE=ELEMENT, NAME={name}\n")
            if nh:
                f.write(f"ES_SC8R, {face}\n")
            if nw:
                f.write(f"ES_SC6R, {face}\n")
            surfaces[name] = face

        # --- orientation par élément ---
        a1, a2 = mesh["axis1"], mesh["axis2"]
        f.write("*DISTRIBUTION TABLE, NAME=ORI_TABLE\n")
        f.write("COORD3D, COORD3D\n")
        f.write("*DISTRIBUTION, NAME=ORI_DIST, LOCATION=ELEMENT, TABLE=ORI_TABLE\n")
        f.write(", 1., 0., 0., 0., 1., 0.\n")
        for k in range(n_el):
            u, v = a1[k], a2[k]
            f.write(f"{k + 1}, {u[0]:.7g}, {u[1]:.7g}, {u[2]:.7g}, {v[0]:.7g}, {v[1]:.7g}, {v[2]:.7g}\n")
        f.write("*ORIENTATION, NAME=ORI_ELEM, DEFINITION=COORDINATES, SYSTEM=RECTANGULAR\n")
        f.write("ORI_DIST\n")
        f.write("3, 0.\n")

        # --- sections : une par zone d'épaisseur ---
        for z in used:
            f.write(f"*SHELL SECTION, ELSET={zone_names[z]}, MATERIAL=TBD, ORIENTATION=ORI_ELEM, "
                    f"STACK DIRECTION=3\n")
            f.write(f"{zone_t[z]:.6g}, 5\n")
    Path(path).write_text(to_ascii(f.getvalue()), encoding="ascii")
    names = [zone_names.get(z, "") for z in range(max(used) + 1)] if used else []
    return dict(zones={zone_names[z]: zone_t[z] for z in used}, elsets=elsets, nsets=nsets,
                zone_of=zone_of, zone_names=names, surfaces=surfaces,
                skin_areas=dict(S1=a_ref, S2=a_opp, rel_diff=rel,
                                inner_outer_significant=bool(rel >= 0.005)))


def write_inp_tet(path: Path, mesh: dict, pa, metrics: dict, status: str) -> dict:
    """Export tétraédrique : C3D10 (ou C3D4), une *SOLID SECTION, pas de matériau."""
    X, C = mesh["nodes"], mesh["conn"]
    etype = "C3D10" if C.shape[1] == 10 else "C3D4"
    f = io.StringIO()
    f.write("*HEADING\n")
    f.write(f"stepmesher {__version__} - {Path(pa.source).name} - {status}\n")
    f.write(f"** genere le {dt.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"** source : {pa.source}\n")
    f.write(f"** repli tetraedrique : {len(C)} {etype}, taille {metrics.get('size', 0):.4g} mm\n")
    f.write(f"** nature detectee : {pa.kind}\n")
    f.write("** MATERIAL=TBD : bloc materiau a fournir\n")
    f.write("*NODE, NSET=NALL\n")
    for i, p in enumerate(X, 1):
        f.write(f"{i}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}\n")
    f.write(f"*ELEMENT, TYPE={etype}, ELSET=ES_ALL\n")
    for k, c in enumerate(C, 1):
        ids = [str(int(x) + 1) for x in c]
        # 16 entrées max par ligne Abaqus : C3D10 = 11 valeurs, tient sur une ligne
        f.write(f"{k}, " + ", ".join(ids) + "\n")
    skin = np.asarray(mesh.get("skin", np.zeros(0, int)), int)
    if len(skin):
        f.write("*NSET, NSET=NS_SKIN\n")
        _ids(f, skin + 1)
    q = mesh.get("quality")
    if q is not None and np.any(q < 0.1):
        f.write("** tetraedres de forme mediocre (qualite < 0.1)\n")
        f.write("*ELSET, ELSET=ES_QUALITY_WARN\n")
        _ids(f, np.nonzero(q < 0.1)[0] + 1)
    f.write("*SOLID SECTION, ELSET=ES_ALL, MATERIAL=TBD\n")
    Path(path).write_text(to_ascii(f.getvalue()), encoding="ascii")
    return dict(element_type=etype)
