"""Relecture et contrôle d'un .inp produit (sans Abaqus).

Vérifie la cohérence interne : numérotation, références de nœuds, connectivité,
volumes positifs, sets non vides, références croisées (surfaces, distribution,
orientation, sections), ordre des mots-clés. Ne remplace pas un datacheck Abaqus.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..mesh.quality import HEX_CORNERS, WEDGE_CORNERS, scaled_jacobian

NNODES = {"SC8R": 8, "SC6R": 6, "C3D8": 8, "C3D6": 6, "C3D4": 4, "C3D10": 10}


def _params(line: str):
    parts = [p.strip() for p in line[1:].split(",")]
    kw = parts[0].upper()
    prm = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            prm[k.strip().upper()] = v.strip()
        elif p:
            prm[p.upper()] = True
    return kw, prm


def read_inp(path: Path) -> dict:
    blocks = []
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw, prm = _params(line)
                cur = dict(kw=kw, prm=prm, data=[])
                blocks.append(cur)
            elif cur is not None:
                cur["data"].append(line)
    return dict(blocks=blocks)


def validate_inp(path: Path) -> dict:
    errors, warnings = [], []
    blocks = read_inp(path)["blocks"]
    nodes = {}
    elems = {}           # id -> (type, conn)
    nsets = defaultdict(set)
    elsets = defaultdict(set)
    surfaces, distributions, orientations, sections = {}, {}, {}, []
    order = []
    for b in blocks:
        kw, prm, data = b["kw"], b["prm"], b["data"]
        order.append(kw)
        if kw == "NODE":
            for ln in data:
                v = [x.strip() for x in ln.split(",")]
                i = int(v[0])
                if i in nodes:
                    errors.append(f"nœud {i} défini deux fois")
                nodes[i] = [float(x) for x in v[1:4]]
            if "NSET" in prm:
                nsets[prm["NSET"].upper()].update(int(ln.split(",")[0]) for ln in data)
        elif kw == "ELEMENT":
            et = prm.get("TYPE", "").upper()
            nn = NNODES.get(et)
            if nn is None:
                errors.append(f"type d'élément non géré : {et}")
                continue
            ids = []
            for ln in data:
                v = [int(x) for x in ln.replace(" ", "").split(",") if x]
                i, conn = v[0], v[1:]
                if i in elems:
                    errors.append(f"élément {i} défini deux fois")
                if len(conn) != nn:
                    errors.append(f"élément {i} ({et}) : {len(conn)} nœuds au lieu de {nn}")
                elif len(set(conn)) != nn:
                    errors.append(f"élément {i} : nœuds dupliqués {conn}")
                elems[i] = (et, conn)
                ids.append(i)
            if "ELSET" in prm:
                elsets[prm["ELSET"].upper()].update(ids)
        elif kw in ("NSET", "ELSET"):
            name = prm[kw].upper()
            target = nsets if kw == "NSET" else elsets
            vals = [x.strip() for ln in data for x in ln.split(",") if x.strip()]
            if "GENERATE" in prm:
                a, b_, s = (int(x) for x in vals[:3])
                target[name].update(range(a, b_ + 1, s))
            else:
                for x in vals:
                    if x.lstrip("-").isdigit():
                        target[name].add(int(x))
                    else:
                        target[name].update(elsets.get(x.upper(), set()) if kw == "ELSET" else nsets.get(x.upper(), set()))
                        if kw == "ELSET" and x.upper() not in elsets:
                            errors.append(f"ELSET {name} référence un set inconnu {x}")
        elif kw == "SURFACE":
            name = prm.get("NAME", "").upper()
            refs = [(ln.split(",")[0].strip().upper(), ln.split(",")[1].strip().upper()) for ln in data]
            surfaces[name] = refs
            for es, face in refs:
                if es not in elsets:
                    errors.append(f"surface {name} : ELSET inconnu {es}")
        elif kw == "DISTRIBUTION":
            name = prm.get("NAME", "").upper()
            rows = {}
            for ln in data:
                v = [x.strip() for x in ln.split(",")]
                if v[0] == "":
                    continue
                rows[int(v[0])] = [float(x) for x in v[1:]]
            distributions[name] = rows
        elif kw == "ORIENTATION":
            orientations[prm.get("NAME", "").upper()] = data[0].strip().upper() if data else ""
        elif kw in ("SHELL SECTION", "SOLID SECTION"):
            sections.append((kw, prm, data))

    # --- contrôles croisés ---
    all_node_ids = set(nodes)
    for i, (et, conn) in elems.items():
        miss = [c for c in conn if c not in all_node_ids]
        if miss:
            errors.append(f"élément {i} : nœuds inexistants {miss[:4]}")
    for name, ids in list(nsets.items()) + list(elsets.items()):
        if not ids:
            errors.append(f"set vide : {name}")
    for name, ids in nsets.items():
        if ids - all_node_ids:
            errors.append(f"NSET {name} : {len(ids - all_node_ids)} nœud(s) inexistant(s)")
    for name, ids in elsets.items():
        if ids - set(elems):
            errors.append(f"ELSET {name} : {len(ids - set(elems))} élément(s) inexistant(s)")
    for o, dist in orientations.items():
        if dist not in distributions:
            errors.append(f"orientation {o} : distribution inconnue {dist}")
    covered = set()
    for kw, prm, data in sections:
        es = prm.get("ELSET", "").upper()
        if es not in elsets:
            errors.append(f"{kw} : ELSET inconnu {es}")
            continue
        if es in covered or (covered & elsets[es]):
            errors.append(f"{kw} {es} : éléments déjà couverts par une autre section")
        covered |= elsets[es]
        ori = prm.get("ORIENTATION", "").upper()
        if ori:
            if ori not in orientations:
                errors.append(f"{kw} {es} : orientation inconnue {ori}")
            else:
                rows = distributions.get(orientations[ori], {})
                missing = elsets[es] - set(rows)
                if missing:
                    errors.append(f"{kw} {es} : {len(missing)} élément(s) sans orientation dans la distribution")
        if prm.get("MATERIAL", "").upper() == "TBD":
            warnings.append(f"{kw} {es} : MATERIAL=TBD (matériau à définir)")
        if kw == "SHELL SECTION":
            try:
                t = float(data[0].split(",")[0])
                if t <= 0:
                    errors.append(f"{kw} {es} : épaisseur <= 0")
            except (IndexError, ValueError):
                errors.append(f"{kw} {es} : ligne de données invalide")
    if set(elems) - covered:
        errors.append(f"{len(set(elems) - covered)} élément(s) sans section")
    for name, rows in distributions.items():
        for i, v in list(rows.items())[:100000]:
            if len(v) != 6:
                errors.append(f"distribution {name} élément {i} : {len(v)} valeurs au lieu de 6")
                break
            a, b = np.array(v[:3]), np.array(v[3:])
            if np.linalg.norm(np.cross(a, b)) < 1e-6:
                errors.append(f"distribution {name} élément {i} : axes colinéaires")
                break
    # ordre : NODE avant ELEMENT avant sets/sections
    if "NODE" in order and "ELEMENT" in order and order.index("NODE") > order.index("ELEMENT"):
        errors.append("*NODE doit précéder *ELEMENT")

    # --- volumes (jacobien normalisé) recalculés depuis le fichier ---
    stats = dict(n_nodes=len(nodes), n_elements=len(elems))
    ids_sorted = sorted(nodes)
    pos = {i: k for k, i in enumerate(ids_sorted)}
    X = np.array([nodes[i] for i in ids_sorted]) if nodes else np.zeros((0, 3))
    for et, corners in (("SC8R", HEX_CORNERS), ("SC6R", WEDGE_CORNERS)):
        conn = np.array([[pos[c] for c in cn] for (t, cn) in elems.values() if t == et
                         and all(c in pos for c in cn)], dtype=np.int64).reshape(-1, NNODES[et])
        if len(conn):
            sj = scaled_jacobian(X, conn, corners)
            stats[f"{et}_min_scaled_jacobian"] = float(sj.min())
            if (sj <= 0).any():
                errors.append(f"{int((sj <= 0).sum())} élément(s) {et} de volume négatif ou nul")
    for et in ("C3D10", "C3D4"):
        conn = np.array([[pos[c] for c in cn[:4]] for (t, cn) in elems.values() if t == et
                         and all(c in pos for c in cn)], dtype=np.int64).reshape(-1, 4)
        if len(conn):
            a, b, c, d = (X[conn[:, i]] for i in range(4))
            vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
            stats[f"{et}_min_volume"] = float(vol.min() / 6)
            if (vol <= 0).any():
                errors.append(f"{int((vol <= 0).sum())} élément(s) {et} de volume négatif ou nul")
    stats.update(n_nsets=len(nsets), n_elsets=len(elsets), n_surfaces=len(surfaces), n_sections=len(sections))
    return dict(ok=not errors, errors=errors[:50], n_errors=len(errors), warnings=warnings[:20], stats=stats)


def read_mesh(path: Path) -> dict:
    """Nœuds et éléments d'un .inp (pour les tests d'aller-retour)."""
    blocks = read_inp(path)["blocks"]
    nodes, elems = {}, {}
    for b in blocks:
        if b["kw"] == "NODE":
            for ln in b["data"]:
                v = ln.split(",")
                nodes[int(v[0])] = [float(x) for x in v[1:4]]
        elif b["kw"] == "ELEMENT":
            for ln in b["data"]:
                v = [int(x) for x in ln.replace(" ", "").split(",") if x]
                elems[v[0]] = (b["prm"]["TYPE"].upper(), v[1:])
    return dict(nodes=nodes, elems=elems)
