"""Maillage quad de la peau de référence.

Stratégies (déterministes) :
- conform  : Frontal-Delaunay for quads + blossom full-quad, conforme aux arêtes CAD.
             Les faces périodiques (cylindres, tores... refusées par le full-quad)
             sont reparamétrées seules en surface composite : toutes les arêtes CAD
             restent des contraintes.
- compound : faces tangentes (angle < tangent_angle) fusionnées en surfaces
             composites : le mailleur ignore les découpages CATIA parasites.
- blossom  : comme conform, recombinaison blossom simple (peut laisser des triangles).
- subdiv   : triangles puis subdivision : 100 % quads garanti, qualité moindre.
- qqs      : quasi-structuré gmsh (algo 11), lent, optionnel.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

PERIODIC_TYPES = ("Cylinder", "Cone", "Torus", "Sphere", "Revolution", "SurfaceOfRevolution")


def setup_reference_model(pa) -> None:
    """Relit la géométrie préparée et ne garde que les faces de la peau de référence.
    Topologie virtuelle (pa.skin_brep) : le modèle de peau à contours simplifiés est chargé
    à la place (il ne contient que les faces de peau) ; numérotation propre, voir skin_view."""
    if getattr(pa, "skin_brep", ""):
        gmsh.model.occ.importShapes(pa.skin_brep)
        gmsh.model.occ.synchronize()
        return
    gmsh.model.occ.importShapes(pa.brep)
    gmsh.model.occ.synchronize()
    ref = set(pa.ref_faces)
    gmsh.model.removeEntities(gmsh.model.getEntities(3))
    others = [(2, f) for _, f in gmsh.model.getEntities(2) if f not in ref]
    if others:
        gmsh.model.removeEntities(others, recursive=True)


def skin_view(pa):
    """Analyse vue dans la numérotation du modèle de peau (après setup_reference_model).

    Sans topologie virtuelle : (pa, None, None). Sinon (pm, n2o, o2n) : copie de pa où les
    faces de peau portent leur numéro dans le modèle de peau, leurs courbes celles de ce modèle,
    et les faces hors peau (chants, peau opposée, parois de trous) un numéro NÉGATIF (-ancien),
    sans collision possible, avec leurs courbes traduites (courbes fusionnées -> courbe neuve).
    n2o / o2n : faces du modèle de peau <-> faces de pa.brep. Tranches fusionnées
    (pa.skin_face_groups) : n2o donne le représentant, o2n envoie chaque tranche sur la face
    fusionnée (quads réattribués à leur vraie face par attempt.reclassify_groups)."""
    if not getattr(pa, "skin_brep", ""):
        return pa, None, None
    import copy
    n2o = {int(k): int(v) for k, v in pa.skin_face_map.items()}
    o2n = {v: k for k, v in n2o.items()}
    for rep, members in (getattr(pa, "skin_face_groups", None) or {}).items():
        for t in members:
            o2n[int(t)] = o2n[int(rep)]
    cmap = {int(k): int(v) for k, v in pa.skin_curve_map.items()}

    def fo(f):
        return o2n.get(int(f), -int(f))

    pm = copy.copy(pa)
    pm.ref_faces = sorted({o2n[f] for f in pa.ref_faces})
    fc = {n: sorted({abs(c) for _, c in gmsh.model.getBoundary([(2, n)], oriented=False)}) for n in n2o}
    for f, cs in pa.face_curves.items():
        if int(f) not in o2n:
            fc[-int(f)] = sorted({cmap[c] for c in cs if c in cmap})
    pm.face_curves = fc
    pm.flank_faces = [fo(f) for f in pa.flank_faces]
    pm.opp_faces = [fo(f) for f in pa.opp_faces]
    pm.failed_faces = [fo(f) for f in pa.failed_faces]
    pm.bends = [dict(b, faces=list(dict.fromkeys(o2n[f] for f in b["faces"] if f in o2n))) for b in pa.bends]
    pm.face_thickness = {fo(f): t for f, t in pa.face_thickness.items()}
    pm.face_cad_sign = {fo(f): v for f, v in pa.face_cad_sign.items()}
    pm.face_adjacency = [[fo(a), fo(b), ang] for a, b, ang in pa.face_adjacency]
    pm.holes_kept = [dict(h, faces=[fo(w) for w in h["faces"]]) for h in pa.holes_kept]
    # coutures : courbes du modèle de peau (une couture dont une courbe n'a pas de
    # correspondance est abandonnée : la fissure sera vue par le critère de continuité)
    pm.seams = [dict(sm, a=list(dict.fromkeys(cmap[c] for c in sm["a"])),
                     b=list(dict.fromkeys(cmap[c] for c in sm["b"])))
                for sm in (getattr(pa, "seams", None) or [])
                if all(c in cmap for c in sm["a"] + sm["b"])]
    return pm, n2o, o2n


def _is_periodic(f: int) -> bool:
    t = gmsh.model.getType(2, f)
    return any(k in t for k in PERIODIC_TYPES)


def tangent_groups(pa, angle_deg: float, exclude: set[int] = frozenset()) -> list[list[int]]:
    ref = set(pa.ref_faces) - set(exclude)
    parent = {f: f for f in ref}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, ang in pa.face_adjacency:
        if a in ref and b in ref and ang < angle_deg:
            parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for f in sorted(ref):
        groups.setdefault(find(f), []).append(f)
    return list(groups.values())


def _even(n: int) -> int:
    return max(2, n + (n % 2))


def _chain_path(curves):
    """Courbes d'une chaîne ouverte dans l'ordre de parcours : [(courbe, sommet de départ,
    sommet d'arrivée)], ou None (courbe absente du modèle, chaîne fermée ou ramifiée)."""
    ep = {}
    for c in curves:
        pts = [abs(t) for _, t in gmsh.model.getBoundary([(1, c)], oriented=False)]
        if len(pts) != 2 or pts[0] == pts[1]:
            return None
        ep[c] = pts
    deg: dict[int, int] = {}
    for pts in ep.values():
        for p in pts:
            deg[p] = deg.get(p, 0) + 1
    tips = sorted(p for p, n in deg.items() if n == 1)
    if len(tips) != 2 or any(n > 2 for n in deg.values()):
        return None
    out, cur, left = [], tips[0], set(curves)
    while left:
        c = next((c for c in left if cur in ep[c]), None)
        if c is None:
            return None
        nxt = ep[c][1] if ep[c][0] == cur else ep[c][0]
        out.append((int(c), int(cur), int(nxt)))
        left.remove(c)
        cur = nxt
    return out


def _short_link(p: int, q: int, max_len: float, max_curves: int = 3) -> list[int]:
    """Courbes courtes (<= max_len) reliant directement les sommets p et q (au plus max_curves
    courbes bout à bout), [] s'il n'y en a pas : bout d'une face-lanière resté dans le modèle
    parce qu'il borde une autre face de la peau."""
    if p == q:
        return []
    seen, front = {p: []}, [p]
    for _ in range(max_curves):
        nxt = []
        for v in front:
            for c in gmsh.model.getAdjacencies(0, v)[0]:
                c = int(c)
                if gmsh.model.occ.getMass(1, c) > max_len:
                    continue
                for _, w in gmsh.model.getBoundary([(1, c)], oriented=False):
                    w = abs(w)
                    if w not in seen:
                        seen[w] = seen[v] + [c]
                        nxt.append(w)
        if q in seen:
            return seen[q]
        front = nxt
    return []


def seam_chains(pa) -> list[dict]:
    """Coutures (pa.seams, faces-lanières retirées de la peau) dans le modèle courant : deux
    chaînes ordonnées et parcourues dans le MÊME sens. [dict(face, a, b, links)], a et b =
    [(courbe, sommet de départ, sommet d'arrivée)] ; links = (courbes du bout de la lanière
    encore présentes au début, à la fin) : elles se réduisent à un point à la soudure.
    Couture ignorée si une chaîne ne se retrouve pas dans le modèle (attempt : échec explicite)."""
    out = []
    for sm in getattr(pa, "seams", None) or []:
        try:
            a, b = _chain_path(list(sm["a"])), _chain_path(list(sm["b"]))
            if a is None or b is None:
                continue
            P = {p: np.array(gmsh.model.getValue(0, p, []), float) for p in (a[0][1], a[-1][2], b[0][1], b[-1][2])}
            same = np.linalg.norm(P[a[0][1]] - P[b[0][1]]) + np.linalg.norm(P[a[-1][2]] - P[b[-1][2]])
            cross = np.linalg.norm(P[a[0][1]] - P[b[-1][2]]) + np.linalg.norm(P[a[-1][2]] - P[b[0][1]])
            if cross < same:
                b = [(c, p1, p0) for c, p0, p1 in reversed(b)]
            lim = max(1.0, 5.0 * float(sm.get("width", 0.0)))
            links = (_short_link(a[0][1], b[0][1], lim), _short_link(a[-1][2], b[-1][2], lim))
            out.append(dict(face=int(sm.get("face", 0)), a=a, b=b, links=links))
        except Exception:  # noqa: BLE001
            continue
    return out


def seam_counts(seams: list[dict], h: float, fixed: dict) -> list[int]:
    """Mêmes nombres de segments (pairs) sur les deux chaînes de chaque couture, hors
    planification harmonisée (plis libres, ancien enchaînement). Les courbes déjà imposées
    (fixed : courbe -> nombre de nœuds) sont respectées. Renvoie les faces des coutures réglées."""
    done = []
    for sm in seams:
        ca, cb = [c for c, _, _ in sm["a"]], [c for c, _, _ in sm["b"]]
        try:
            L = {c: gmsh.model.occ.getMass(1, c) for c in ca + cb}
            cnt = {c: fixed[c] - 1 if c in fixed else _even(int(round(L[c] / h))) for c in ca + cb}
            ta, tb = sum(cnt[c] for c in ca), sum(cnt[c] for c in cb)
            if ta != tb:
                cand = [c for c in (ca if ta < tb else cb) if c not in fixed]
                if not cand:
                    continue
                cnt[max(cand, key=lambda c: L[c])] += abs(ta - tb)
            for c in ca + cb:
                if c not in fixed:
                    gmsh.model.mesh.setTransfiniteCurve(c, cnt[c] + 1)
                    fixed[c] = cnt[c] + 1
            done.append(sm["face"])
        except Exception:  # noqa: BLE001
            continue
    return done


def _split_count(lengths: list[float], total: int, even=None) -> list[int]:
    """Répartit `total` segments sur une chaîne de courbes, au prorata des longueurs (>= 1 chacune).

    even : courbes devant porter un nombre PAIR de segments (partagées avec une face libre
    maillée en full-quad : une micro-courbe à 1 segment y fait échouer la face voisine,
    « 1D mesh cannot be divided by 2 », part_014). La somme peut alors différer de `total`
    si c'est impossible : à l'appelant de vérifier."""
    L = np.asarray(lengths, float)
    step = np.where(np.asarray(even, bool), 2, 1) if even is not None else np.ones(len(L), int)
    if len(L) == 1:
        return [total]
    target = total * L / L.sum()
    raw = np.maximum(step, np.floor(target / step).astype(int) * step)
    while raw.sum() > total:
        over = np.where(raw - step >= step, raw - target, -np.inf)
        if not np.isfinite(over).any():
            break
        i = int(np.argmax(over))
        raw[i] -= step[i]
    while raw.sum() < total:
        miss = total - raw.sum()
        under = np.where(step <= miss, target - raw, -np.inf)
        if not np.isfinite(under).any():
            break
        i = int(np.argmax(under))
        raw[i] += step[i]
    return raw.tolist()


def _ordered_loop(f: int):
    """Courbes du contour d'une face (une seule boucle) dans l'ordre de parcours,
    avec (point de départ, point d'arrivée) de chacune. Ni getCurveLoops ni
    getBoundary ne garantissent cet ordre."""
    curves = [abs(c) for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)]
    ep = {}
    for c in curves:
        pts = [abs(t) for _, t in gmsh.model.getBoundary([(1, c)], oriented=False)]
        if len(pts) != 2 or pts[0] == pts[1]:
            return None, None
        ep[c] = pts
    order, ends = [curves[0]], [tuple(ep[curves[0]])]
    left = set(curves[1:])
    while left:
        cur = ends[-1][1]
        nxt = next((c for c in left if cur in ep[c]), None)
        if nxt is None:
            return None, None
        a, b = ep[nxt]
        ends.append((a, b) if a == cur else (b, a))
        order.append(nxt)
        left.remove(nxt)
    if ends[-1][1] != ends[0][0]:
        return None, None
    return order, ends


def _curve_turn_deg(c: int) -> float:
    """Rotation de la tangente entre les deux bouts d'une courbe (0 = droite)."""
    try:
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        t0 = np.array(gmsh.model.getDerivative(1, c, [lo[0]]))
        t1 = np.array(gmsh.model.getDerivative(1, c, [hi[0]]))
        cs = float(t0 @ t1) / max(np.linalg.norm(t0) * np.linalg.norm(t1), 1e-300)
        return float(np.degrees(np.arccos(np.clip(cs, -1.0, 1.0))))
    except Exception:  # noqa: BLE001
        return 0.0


def _bend_sides(f: int, by_turn: bool = False):
    """Contour d'une face de pli découpé en 4 côtés : (génératrice, arcs, génératrice, arcs).

    Les deux génératrices sont les deux plus longues courbes, non adjacentes ; les
    chaînes d'arcs sont ce qui reste entre elles. Retourne (côtés, coins) ou None.
    by_turn (plis) : génératrices = les deux plus longues courbes DROITES (rotation de
    tangente < 10°) ; une tranche de pli plus courte que son arc (pli découpé le long de
    l'axe, cadres : arc 9,8 mm, tranche 1,75 mm) prenait sinon ses arcs pour génératrices
    -> conflit avec le pli voisin, face libre, triangles et hexa gauchis (upper part_022).
    """
    loops, _ = gmsh.model.occ.getCurveLoops(f)
    if len(loops) != 1:
        return None
    loop, ends = _ordered_loop(f)
    if loop is None:
        return None
    n = len(loop)
    if n < 4:
        return None
    L = [gmsh.model.occ.getMass(1, abs(c)) for c in loop]
    order = sorted(range(n), key=lambda i: -L[i])
    if by_turn:
        straight = [i for i in order if _curve_turn_deg(abs(loop[i])) < 10.0]
        pair = next(((a, b) for k, a in enumerate(straight) for b in straight[k + 1:]
                     if abs(a - b) not in (1, n - 1)), None)
        if pair is not None:
            order = list(pair)
    i1, i2 = sorted(order[:2])
    if (i2 - i1) in (1, n - 1):
        return None
    chain_a = list(range(i1 + 1, i2))
    chain_b = list(range(i2 + 1, n)) + list(range(0, i1))
    if not chain_a or not chain_b:
        return None

    corners = [ends[i1][0], ends[chain_a[0]][0], ends[i2][0], ends[chain_b[0]][0]]
    if len(set(corners)) != 4:
        return None
    sides = [[abs(loop[i1])], [abs(loop[i]) for i in chain_a], [abs(loop[i2])], [abs(loop[i]) for i in chain_b]]
    lens = [[L[i1]], [L[i] for i in chain_a], [L[i2]], [L[i] for i in chain_b]]
    return sides, lens, corners


def structured_bends(pa, recipe, h0: float, size_factor: float = 1.0, fixed: dict | None = None,
                     max_bend_angle_deg: float = 15.0, min_bend_size_frac: float = 0.0,
                     angle_floor: bool = False) -> list[int]:
    """Plis maillés en transfini : n_per_bend éléments sur l'arc, alignés sur le pli.

    Accepte les faces à 4 côtés et celles dont les arcs sont découpés en plusieurs
    courbes (exports CATIA). Les autres restent en maillage libre.
    """
    done = []
    fixed = {} if fixed is None else fixed
    free = set(getattr(recipe, "free_faces", ()) or ())
    n_ref_faces: dict[int, int] = {}
    for g in pa.ref_faces:
        for c in pa.face_curves.get(g, []):
            n_ref_faces[abs(c)] = n_ref_faces.get(abs(c), 0) + 1
    # passe 1 : côtés et nombre d'éléments sur l'arc de chaque face de pli
    plans = []
    for b in pa.bends:
        for f in b["faces"]:
            if f in free:
                continue
            try:
                r = _bend_sides(f, by_turn=True)
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                continue
            sides, lens, corners = r
            arc_len = max(sum(lens[1]), sum(lens[3]))
            gen_len = min(lens[0][0], lens[2][0])
            # les arcs doivent être nettement plus courts que les génératrices
            if arc_len > 0.8 * gen_len and arc_len > 2 * h0:
                continue
            # au moins n_per_bend éléments, et au plus max_bend_angle_deg par élément
            face_angle = np.degrees(arc_len / max(b["radius"], 1e-9))
            n_ang = int(np.ceil(face_angle / max(max_bend_angle_deg, 1e-6) - 1e-6))
            n_arc = max(1, int(round(max(recipe.n_per_bend, n_ang) / size_factor)))
            n_arc = max(n_arc, len(sides[1]), len(sides[3]))
            # parité imposée seulement si un arc borde une autre face de la peau
            # (contour maillé librement : nombre de segments pair pour des quads purs) ;
            # sur un bord libre (arc posé sur un chant), 3 éléments sur 90° sont permis
            shared = any(n_ref_faces.get(c, 0) > 1 for c in sides[1] + sides[3])
            if shared:
                n_arc += n_arc % 2
            if min_bend_size_frac > 0:
                # ne pas descendre sous min_bend_size_frac * h0, même si l'arc est petit
                # (sinon un pli à petit rayon impose une bande d'éléments minuscules sur
                # toute sa longueur dans une pièce par ailleurs grossière) : le plafond
                # est arrondi au pair INFÉRIEUR, sinon la parité l'annule silencieusement
                floor = max(len(sides[1]), len(sides[3]), 2 if shared else 1)
                # règle de Fred : jamais moins d'1 élément par max_bend_angle_deg (3 sur 90°)
                if angle_floor:
                    floor = max(floor, n_ang + (n_ang % 2 if shared else 0))
                cap = max(floor, int(arc_len / (min_bend_size_frac * h0 * size_factor)))
                cap = max(floor, cap - cap % 2 if shared else cap)
                n_arc = min(n_arc, cap)
            plans.append((f, sides, lens, corners, arc_len, n_arc))
    if not plans:
        return done
    # densité commune le long des génératrices (élancement <= 6 dans le pli le plus
    # serré) : deux génératrices de même longueur reçoivent le même nombre de nœuds,
    # ce qui évite les aiguilles dans les bandes étroites entre deux plis
    h_gen = min([h0 * size_factor] + [6.0 * a / n for _, _, _, _, a, n in plans])
    for f, sides, lens, corners, arc_len, n_arc in plans:
        n_gen = _even(int(round(max(lens[0][0], lens[2][0]) / h_gen)))
        # chaîne d'arcs découpée : chaque courbe partagée avec une autre face de peau reçoit
        # un nombre pair de segments (sinon n_arc + 1, + 2 ; face laissée libre au-delà)
        ev = [[n_ref_faces.get(c, 0) > 1 for c in sides[k]] for k in range(4)]
        splits = None
        for tot_arc in (n_arc, n_arc + 1, n_arc + 2):
            sa = _split_count(lens[1], tot_arc, ev[1])
            sb = _split_count(lens[3], tot_arc, ev[3])
            if sum(sa) == tot_arc and sum(sb) == tot_arc:
                splits = (sa, sb)
                break
        if splits is None:
            continue
        want = {}
        for side, ln, cnt in ((sides[0], lens[0], [n_gen]), (sides[1], lens[1], splits[0]),
                              (sides[2], lens[2], [n_gen]), (sides[3], lens[3], splits[1])):
            for c, k in zip(side, cnt):
                want[c] = k + 1
        if any(c in fixed and fixed[c] != n for c, n in want.items()):
            continue
        try:
            for c, n in want.items():
                gmsh.model.mesh.setTransfiniteCurve(c, n)
            if sum(len(x) for x in sides) == 4:
                gmsh.model.mesh.setTransfiniteSurface(f)
            else:
                gmsh.model.mesh.setTransfiniteSurface(f, cornerTags=corners)
            gmsh.model.mesh.setRecombine(2, f)
            fixed.update(want)
            done.append(f)
        except Exception:  # noqa: BLE001
            continue
    return done


def _curve_total_turn_deg(c: int, k: int = 8) -> float:
    """Rotation totale de la tangente le long d'une courbe (somme sur k intervalles)."""
    try:
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        ts = np.linspace(lo[0], hi[0], k + 1)
        D = np.array(gmsh.model.getDerivative(1, c, ts.tolist())).reshape(-1, 3)
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-300)
        cs = np.clip(np.einsum("ij,ij->i", D[:-1], D[1:]), -1.0, 1.0)
        return float(np.degrees(np.arccos(cs)).sum())
    except Exception:  # noqa: BLE001
        return 0.0


def _traversal_tangents(c: int, start_pt: int) -> tuple[np.ndarray, np.ndarray]:
    """Tangentes unitaires (au départ, à l'arrivée) d'une courbe parcourue depuis le
    sommet start_pt (sens de parcours de la boucle, pas forcément celui du paramétrage)."""
    lo, hi = gmsh.model.getParametrizationBounds(1, c)
    d = np.array(gmsh.model.getDerivative(1, c, [lo[0], hi[0]]), float).reshape(2, 3)
    p0 = np.array(gmsh.model.getValue(1, c, [lo[0]]), float)
    ps = np.array(gmsh.model.getValue(0, start_pt, []), float)
    pe = np.array(gmsh.model.getValue(1, c, [hi[0]]), float)
    if np.linalg.norm(p0 - ps) > np.linalg.norm(pe - ps):      # parcours à rebours
        d = -d[::-1]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    return d[0], d[1]


def face_outline(f: int, short_len: float, corner_deg: float = 30.0) -> dict | None:
    """Coins réels du contour d'une face à une seule boucle : virage >= corner_deg entre
    deux courbes « longues » (>= short_len) consécutives, les micro-courbes intermédiaires
    (chanfrein de 1,9 mm au bout d'une patte) étant sautées. 30° : une patte coupée en biais
    (upper part_001) a des coins obtus à ~40° de virage ; les raccords entre morceaux d'un
    même côté font < 1°. Renvoie
    dict(corners=n, sides=[longueur de chaque chaîne entre coins]) ou None."""
    loops, _ = gmsh.model.occ.getCurveLoops(f)
    if len(loops) != 1:
        return None
    loop, ends = _ordered_loop(f)
    if loop is None:
        return None
    L = [gmsh.model.occ.getMass(1, abs(c)) for c in loop]
    tans = [_traversal_tangents(abs(c), a) for c, (a, _) in zip(loop, ends)]
    long_idx = [i for i in range(len(loop)) if L[i] >= short_len]
    if len(long_idx) < 4:
        return None
    n = len(long_idx)
    corner_after = []           # coin entre long_idx[k] et long_idx[k+1]
    for k in range(n):
        i, j = long_idx[k], long_idx[(k + 1) % n]
        cs = float(np.clip(tans[i][1] @ tans[j][0], -1.0, 1.0))
        corner_after.append(np.degrees(np.arccos(cs)) >= corner_deg)
    n_corners = int(sum(corner_after))
    sides, acc = [], 0.0
    start = next((k for k in range(n) if corner_after[k]), None)
    if start is None:
        return dict(corners=0, sides=[])
    # chaînes entre coins (longueur totale, micro-courbes comprises)
    for step in range(1, len(loop) + 1):
        i = (long_idx[start] + step) % len(loop)
        acc += L[i]
        if i in long_idx and corner_after[long_idx.index(i)]:
            sides.append(acc)
            acc = 0.0
    return dict(corners=n_corners, sides=sides)


def _rect_sides(f: int, short_len: float, corner_deg: float = 30.0):
    """Face à 4 coins réels (face_outline) découpée en 4 côtés = chaînes de courbes entre coins.

    Contrairement à _bend_sides (2 plus longues courbes = génératrices), un grand côté peut
    être découpé en plusieurs courbes (patte de upper part_001 bordée par un pli en 7
    faces). Les micro-courbes (< short_len) situées à un coin vont du côté que leur sommet de
    plus fort virage désigne (coin réel), jamais un raccord tangent en coin.
    Renvoie (sides, lens, corners, turns) ou None ; turns = virage (deg) à chaque coin."""
    loops, _ = gmsh.model.occ.getCurveLoops(f)
    if len(loops) != 1:
        return None
    loop, ends = _ordered_loop(f)
    if loop is None:
        return None
    m = len(loop)
    L = [gmsh.model.occ.getMass(1, abs(c)) for c in loop]
    tans = [_traversal_tangents(abs(c), a) for c, (a, _) in zip(loop, ends)]
    long_idx = [i for i in range(m) if L[i] >= short_len]
    if len(long_idx) < 4:
        return None
    def turn(a, b):
        return float(np.degrees(np.arccos(np.clip(tans[a][1] @ tans[b][0], -1.0, 1.0))))

    starts, turns = [], []          # indice de la 1re courbe de chaque côté
    n = len(long_idx)
    for k in range(n):
        i, j = long_idx[k], long_idx[(k + 1) % n]
        ang = turn(i, j)
        if ang < corner_deg:
            continue
        # coin = sommet où le contour tourne VRAIMENT parmi i, micro-courbes, j : un sommet de
        # raccord tangent (chanfrein de 1,9 mm prolongeant le bout d'une patte, upper part_001)
        # comme coin donne un quad plat à 180° -> retourné
        chain = [i] + [(i + d) % m for d in range(1, (j - i) % m)] + [j]
        best = max(range(len(chain) - 1), key=lambda t: turn(chain[t], chain[t + 1]))
        starts.append(chain[best + 1])
        turns.append(ang)
    if len(starts) != 4:
        return None
    order = sorted(range(4), key=lambda k: starts[k])
    starts = [starts[k] for k in order]
    turns = [turns[k] for k in order]
    sides, lens = [], []
    for k in range(4):
        a, b = starts[k], starts[(k + 1) % 4]
        idx = list(range(a, b)) if b > a else list(range(a, m)) + list(range(0, b))
        if not idx:
            return None
        sides.append([abs(loop[i]) for i in idx])
        lens.append([L[i] for i in idx])
    corners = [ends[st][0] for st in starts]
    if len(set(corners)) != 4:
        return None
    return sides, lens, corners, turns


def regular_expected_faces(pa, h0: float, corner_deg: float = 30.0, max_side_ratio: float = 2.0) -> dict:
    """Faces de peau qui DEVRAIENT être maillées régulièrement (rangées alignées, aucun nœud
    intérieur à 3 ou 5 éléments) : une seule boucle, 4 coins réels, côtés opposés de
    longueurs voisines (rapport <= max_side_ratio) -> patte, lanière, pli, rectangle.
    Renvoie {face: dict(sides=[...])}."""
    out = {}
    for f in pa.ref_faces:
        try:
            o = face_outline(f, 0.5 * h0, corner_deg)
        except Exception:  # noqa: BLE001
            continue
        if not o or o["corners"] != 4 or len(o["sides"]) != 4:
            continue
        a, b, c, d = o["sides"]
        if max(a, c) > max_side_ratio * max(min(a, c), 1e-9) or max(b, d) > max_side_ratio * max(min(b, d), 1e-9):
            continue
        out[int(f)] = dict(sides=[round(x, 1) for x in o["sides"]])
    return out


def contour_arcs(pa, h0: float, fixed: dict, max_angle_deg: float = 30.0, min_turn_deg: float = 15.0,
                 min_size_frac: float = 0.35, even: bool = True, skip=frozenset()) -> list[int]:
    """Coins arrondis du CONTOUR de la peau (courbe peau/chant qui tourne, vue dans le plan
    de la tôle) : « 3 éléments dans un rayon, pas plus » (Fred) -> au plus 1 élément par
    max_angle_deg, segments d'au moins min_size_frac x h0, et au moins L/h0 segments ; pair
    en full-quad. Sans cela, les sommets CAD rapprochés aux bouts de l'arc et la gradation
    autour des arêtes courtes y mettaient 12 segments de 0,7 mm (part_004, arc R5 ; part_014,
    R6-R9 : 8 à 15 segments sur 90°). R4-R5 à 90° : 2 segments ; R13 : 4. Imposer 1 élément
    par 30° sur un petit rayon (4 x 1,6 mm sur R4, part_016) créait au contraire des rosettes.
    Trous conservés et courbes déjà imposées (faces transfinies) exclus. Renvoie les courbes
    imposées. Arcs plus courts que 0,5 h0 ignorés."""
    ref = set(pa.ref_faces)
    ref_curves = {c for f in ref for c in pa.face_curves.get(f, [])}
    flank_curves = {c for f in pa.flank_faces for c in pa.face_curves.get(f, [])}
    hole_walls = {w for h in pa.holes_kept for w in h["faces"]}
    hole_curves = {c for w in hole_walls for c in pa.face_curves.get(w, [])}
    done = []
    for c in sorted((ref_curves & flank_curves) - hole_curves - set(skip)):
        if c in fixed:
            continue
        turn = _curve_total_turn_deg(c)
        if turn < min_turn_deg:
            continue
        L = gmsh.model.occ.getMass(1, c)
        if L < 0.5 * h0:
            # arc court : laissé à la fusion en composite / à la relaxation du contour (l'imposer
            # l'exclut de la fusion -> gradation autour -> rosette, part_016)
            continue
        n_ang = int(np.ceil(turn / max(max_angle_deg, 1e-6) - 0.05))   # 90,2° -> 3, pas 4
        n = max(int(round(L / h0)), min(n_ang, int(L / max(min_size_frac * h0, 1e-9))), 1)
        if even:
            n += n % 2
        try:
            gmsh.model.mesh.setTransfiniteCurve(c, n + 1)
        except Exception:  # noqa: BLE001
            continue
        fixed[c] = n + 1
        done.append(c)
    return done


def merge_micro_curves(max_len: float, protect: set[int] = frozenset()) -> list[list[int]]:
    """Fusionne chaque micro-courbe (< max_len) avec une courbe voisine bordant les
    mêmes faces, en courbe composite : le sommet commun n'impose plus de nœud.
    Seuls les sommets reliant exactement deux courbes sont supprimables."""
    curves = [c for _, c in gmsh.model.getEntities(1)]
    length = {c: gmsh.model.occ.getMass(1, c) for c in curves}
    faces = {c: tuple(sorted(gmsh.model.getAdjacencies(1, c)[0])) for c in curves}
    ends = {c: [abs(t) for _, t in gmsh.model.getBoundary([(1, c)], oriented=False)] for c in curves}
    at_vertex: dict[int, list[int]] = {}
    for c, vs in ends.items():
        for v in vs:
            at_vertex.setdefault(v, []).append(c)
    used, groups = set(), []
    for c in sorted(curves, key=lambda x: length[x]):
        if length[c] >= max_len or c in used or c in protect:
            continue
        best = None
        for v in ends[c]:
            inc = at_vertex.get(v, [])
            if len(inc) != 2:
                continue
            o = inc[0] if inc[1] == c else inc[1]
            if o in used or o in protect or faces[o] != faces[c]:
                continue
            if best is None or length[o] > length[best]:
                best = o
        if best is not None:
            groups.append([c, best])
            used.update((c, best))
    done = []
    for g in groups:
        try:
            gmsh.model.mesh.setCompound(1, g)
            done.append(g)
        except Exception:  # noqa: BLE001
            pass
    return done


def _corner_angles(loop, ends) -> list[float]:
    """Angles intérieurs approchés aux sommets d'un contour ordonné (tangentes aux extrémités)."""
    def tangent_out(c, v):
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        a, b = lo[0], hi[0]
        pa_ = np.array(gmsh.model.getValue(1, c, [a, a + 0.02 * (b - a)])).reshape(2, 3)
        pb_ = np.array(gmsh.model.getValue(1, c, [b, b - 0.02 * (b - a)])).reshape(2, 3)
        pv = np.array(gmsh.model.getValue(0, v, []))
        seg = pa_ if np.linalg.norm(pa_[0] - pv) < np.linalg.norm(pb_[0] - pv) else pb_
        t = seg[1] - seg[0]
        return t / max(np.linalg.norm(t), 1e-300)
    angs = []
    for i in range(len(loop)):
        v = ends[i][0]
        u = tangent_out(loop[i - 1], v)
        w = tangent_out(loop[i], v)
        angs.append(float(np.degrees(np.arccos(np.clip(u @ w, -1, 1)))))
    return angs


def structured_patches(pa, h0: float, fixed: dict, exclude: set[int], size_factor: float = 1.0,
                       max_side: float = float("inf")) -> list[int]:
    """Faces à 4 côtés, coins entre 45° et 135°, côtés opposés de longueurs voisines,
    plus grand côté <= max_side : maillage transfini (quads réguliers), compatible
    avec les courbes déjà fixées. Limité par défaut aux petites faces, où le
    maillage libre produit des quads très plats."""
    nref: dict[int, int] = {}
    for g in pa.ref_faces:
        for c in pa.face_curves.get(g, []):
            nref[abs(c)] = nref.get(abs(c), 0) + 1
    done = []
    for f in pa.ref_faces:
        if f in exclude:
            continue
        try:
            loops, _ = gmsh.model.occ.getCurveLoops(f)
            if len(loops) != 1:
                continue
            loop, ends = _ordered_loop(f)
            if loop is None or len(loop) < 4:
                continue
            if len(loop) == 4:
                angs = _corner_angles(loop, ends)
                if min(angs) < 45 or max(angs) > 135:
                    continue
                sides = [[c] for c in loop]
                slens = [[gmsh.model.occ.getMass(1, c)] for c in loop]
                corners = None
            else:
                # contour découpé (micro-courbes) : 2 côtés longs + 2 chaînes courtes
                r = _bend_sides(f)
                if r is None:
                    continue
                sides, slens, corners = r
            L = [sum(x) for x in slens]
            if max(L) > max_side:
                continue
            if max(L[0], L[2]) > 1.5 * min(L[0], L[2]) or max(L[1], L[3]) > 1.5 * min(L[1], L[3]):
                continue
            want = {}
            ok = True
            tots, is_fixed = {}, {}
            for a_, b_ in ((0, 2), (1, 3)):
                tot_fixed = []
                for sd in (a_, b_):
                    if all(c in fixed for c in sides[sd]):
                        tot_fixed.append(sum(fixed[c] - 1 for c in sides[sd]))
                    elif any(c in fixed for c in sides[sd]):
                        ok = False
                if not ok or (len(tot_fixed) == 2 and tot_fixed[0] != tot_fixed[1]):
                    ok = False
                    break
                is_fixed[a_] = bool(tot_fixed)
                tots[a_] = tot_fixed[0] if tot_fixed else _even(int(round(max(L[a_], L[b_]) / (h0 * size_factor))))
            if not ok:
                continue
            # face étroite (1,75 x 21 mm entre deux tranches de pli, part_022 upper) : 2 segments
            # imposés en travers -> quads 0,9 x 10,7 (élancement 12). Sens long redécoupé pour
            # un élancement <= 6 quand il n'est pas imposé par un voisin
            seg = {k: max(L[k], L[k + 2]) / max(tots[k], 1) for k in (0, 1)}
            lg, sh = (0, 1) if seg[0] >= seg[1] else (1, 0)
            if seg[lg] > 6.0 * seg[sh] and not is_fixed[lg]:
                tots[lg] = _even(int(np.ceil(max(L[lg], L[lg + 2]) / (6.0 * seg[sh]))))
            for a_, b_ in ((0, 2), (1, 3)):
                tot = tots[a_]
                for sd in (a_, b_):
                    if all(c in fixed for c in sides[sd]):
                        continue
                    cnt = _split_count(slens[sd], tot, [nref.get(c, 0) > 1 for c in sides[sd]])
                    if sum(cnt) != tot:        # parité impossible : face laissée libre
                        ok = False
                        break
                    for c, k in zip(sides[sd], cnt):
                        want[c] = k + 1
                if not ok:
                    break
            if not ok:
                continue
            for c, n in want.items():
                gmsh.model.mesh.setTransfiniteCurve(c, n)
            if corners is None:
                gmsh.model.mesh.setTransfiniteSurface(f)
            else:
                gmsh.model.mesh.setTransfiniteSurface(f, cornerTags=corners)
            gmsh.model.mesh.setRecombine(2, f)
            fixed.update(want)
            done.append(f)
        except Exception:  # noqa: BLE001
            continue
    return done


def structured_strips(pa, recipe, h0: float, fixed: dict, exclude: set[int], size_factor: float = 1.0,
                      max_width_factor: float = 4.0, min_length_ratio: float = 3.0,
                      min_size_frac: float = 0.35) -> list[int]:
    """Lanières (faces longues et étroites : largeur <= max_width_factor x h0, longueur >=
    min_length_ratio x largeur) maillées en rangées régulières (transfini). En maillage
    libre full-quad, chaque triangle orphelin y devient une « étoile » de 3 petits quads.

    Bouts découpés en plusieurs courbes : autant de rangées que de courbes au bout le plus
    découpé, tant que la rangée reste >= min_size_frac x h0 ; sinon la face reste libre.
    Parité : un côté partagé avec une autre face de peau porte un nombre pair de segments
    (contour des faces libres voisines en full-quad)."""
    free = set(getattr(recipe, "free_faces", ()) or ())
    h = h0 * size_factor
    nref: dict[int, int] = {}
    for g in pa.ref_faces:
        for c in pa.face_curves.get(g, []):
            nref[abs(c)] = nref.get(abs(c), 0) + 1

    def shared(cs):
        return any(nref.get(c, 0) > 1 for c in cs)

    done = []
    for f in pa.ref_faces:
        if f in exclude or f in free:
            continue
        try:
            r = _bend_sides(f)
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            continue
        sides, lens, corners = r
        L = (lens[0][0], lens[2][0])
        if max(L) > 1.5 * min(L):
            continue
        Lm = float(np.mean(L))
        w = gmsh.model.occ.getMass(2, f) / Lm
        if w > max_width_factor * h or Lm < min_length_ratio * w:
            continue
        # bouts ~droits en travers : un bout long et découpé (redans, coin arrondi
        # prolongé) tord le transfini (éléments retournés, part_021 face 59)
        if max(sum(lens[1]), sum(lens[3])) > 1.5 * w:
            continue
        # rangées en travers
        n_w = max(1, int(round(w / h)), len(sides[1]), len(sides[3]))
        if w / n_w < min_size_frac * h:
            continue
        ends_shared = shared(sides[1]) or shared(sides[3])
        if ends_shared:
            if len(sides[1]) > 1 and shared(sides[1]) or len(sides[3]) > 1 and shared(sides[3]):
                continue
            n_w += n_w % 2
        # segments le long : imposés par un voisin déjà structuré (pli), sinon h
        fixed_l = {fixed[sides[sd][0]] - 1 for sd in (0, 2) if sides[sd][0] in fixed}
        if len(fixed_l) > 1:
            continue
        if fixed_l:
            n_l = fixed_l.pop()
        else:
            n_l = max(1, int(round(Lm / h)))
            if shared(sides[0]) or shared(sides[2]):
                n_l += n_l % 2
        want = {sides[0][0]: n_l + 1, sides[2][0]: n_l + 1}
        for sd in (1, 3):
            for c, k in zip(sides[sd], _split_count(lens[sd], n_w)):
                want[c] = k + 1
        if any(c in fixed and fixed[c] != n for c, n in want.items()):
            continue
        try:
            for c, n in want.items():
                gmsh.model.mesh.setTransfiniteCurve(c, n)
            gmsh.model.mesh.setTransfiniteSurface(f, cornerTags=corners)
            gmsh.model.mesh.setRecombine(2, f)
        except Exception:  # noqa: BLE001
            continue
        fixed.update(want)
        done.append(f)
    return done


def _bend_plans(pa, recipe, h0: float, size_factor: float, max_bend_angle_deg: float,
                min_bend_size_frac: float, angle_floor: bool, n_ref_faces: dict) -> list[dict]:
    """Plis à structurer (mêmes règles que structured_bends) : côtés, nombre d'éléments
    sur l'arc. Les génératrices sont comptées plus tard (densité commune h_gen)."""
    free = set(getattr(recipe, "free_faces", ()) or ())
    plans = []
    for b in pa.bends:
        for f in b["faces"]:
            if f in free:
                continue
            try:
                r = _bend_sides(f, by_turn=True)
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                continue
            sides, lens, corners = r
            arc_len = max(sum(lens[1]), sum(lens[3]))
            gen_len = min(lens[0][0], lens[2][0])
            if arc_len > 0.8 * gen_len and arc_len > 2 * h0:
                continue
            face_angle = np.degrees(arc_len / max(b["radius"], 1e-9))
            n_ang = int(np.ceil(face_angle / max(max_bend_angle_deg, 1e-6) - 1e-6))
            n_arc = max(1, int(round(max(recipe.n_per_bend, n_ang) / size_factor)))
            n_arc = max(n_arc, len(sides[1]), len(sides[3]))
            shared = any(n_ref_faces.get(c, 0) > 1 for c in sides[1] + sides[3])
            if shared:
                n_arc += n_arc % 2
            if min_bend_size_frac > 0:
                floor = max(len(sides[1]), len(sides[3]), 2 if shared else 1)
                if angle_floor:
                    floor = max(floor, n_ang + (n_ang % 2 if shared else 0))
                cap = max(floor, int(arc_len / (min_bend_size_frac * h0 * size_factor)))
                cap = max(floor, cap - cap % 2 if shared else cap)
                n_arc = min(n_arc, cap)
            plans.append(dict(face=int(f), kind="pli", sides=sides, lens=lens,
                              corners=None if sum(len(x) for x in sides) == 4 else corners,
                              arc_len=arc_len, n_arc=n_arc))
    return plans


def _strip_plan_legacy(f: int, h: float, n_ref_faces: dict, max_width_factor: float,
                       min_length_ratio: float, min_size_frac: float) -> dict | None:
    """Lanière selon l'ancienne règle (structured_strips) : grands côtés = 2 plus longues
    courbes, bouts courts (<= 1,5 x largeur). Pour structured_plan quand _rect_sides échoue."""
    try:
        r = _bend_sides(f)
    except Exception:  # noqa: BLE001
        r = None
    if r is None:
        return None
    sides, lens, corners = r
    L = (lens[0][0], lens[2][0])
    if max(L) > 1.5 * min(L):
        return None
    Lm = float(np.mean(L))
    w = gmsh.model.occ.getMass(2, f) / Lm
    if w > max_width_factor * h or Lm < min_length_ratio * w:
        return None
    if max(sum(lens[1]), sum(lens[3])) > 1.5 * w:
        return None
    ends_shared = any(n_ref_faces.get(c, 0) > 1 for c in sides[1] + sides[3])
    if ends_shared and (len(sides[1]) > 1 or len(sides[3]) > 1):
        return None
    n_w = max(1, int(round(w / h)), len(sides[1]), len(sides[3]))
    if w / n_w < min_size_frac * h:
        return None
    return dict(face=int(f), kind="lanière", sides=sides, lens=lens,
                corners=None if sum(len(x) for x in sides) == 4 else corners,
                target=[max(1, int(round(Lm / h))), n_w])


def _patch_plan_legacy(f: int, h: float, max_side: float) -> dict | None:
    """Petite face à 4 côtés selon l'ancienne règle (structured_patches)."""
    try:
        loops, _ = gmsh.model.occ.getCurveLoops(f)
        if len(loops) != 1:
            return None
        loop, ends = _ordered_loop(f)
        if loop is None or len(loop) < 4:
            return None
        if len(loop) == 4:
            angs = _corner_angles(loop, ends)
            if min(angs) < 45 or max(angs) > 135:
                return None
            sides = [[abs(c)] for c in loop]
            lens = [[gmsh.model.occ.getMass(1, abs(c))] for c in loop]
            corners = None
        else:
            r = _bend_sides(f)
            if r is None:
                return None
            sides, lens, corners = r
    except Exception:  # noqa: BLE001
        return None
    tot = [sum(x) for x in lens]
    if max(tot) > max_side:
        return None
    if max(tot[0], tot[2]) > 1.5 * min(tot[0], tot[2]) or max(tot[1], tot[3]) > 1.5 * min(tot[1], tot[3]):
        return None
    tgt = [_even(int(round(max(tot[k], tot[k + 2]) / h))) for k in (0, 1)]
    return dict(face=int(f), kind="patch", sides=sides, lens=lens, corners=corners, target=tgt)


def structured_plan(pa, recipe, h0: float, size_factor: float, fixed: dict,
                    max_bend_angle_deg: float = 30.0, min_bend_size_frac: float = 0.0,
                    angle_floor: bool = False, strips_on: bool = True, patch_max_side: float | None = None,
                    max_width_factor: float = 4.0, min_length_ratio: float = 3.0, min_size_frac: float = 0.35,
                    max_sweeps: int = 400, seams: list | None = None) -> dict:
    """Faces structurées (transfini) planifiées ENSEMBLE : plis, lanières, petites faces à 4 côtés.

    L'ancien enchaînement face par face (plis, puis lanières, puis petites faces, face ignorée
    au premier conflit de nombre de segments) laissait en maillage libre les faces bordées
    par des transfinis voisins aux comptes incompatibles : patte de 7,4 m x 36 mm de upper
    part_001, d'un côté un pli d'une seule face (7 328 mm), de l'autre un pli découpé en 7
    faces arrondies chacune -> totaux différents -> maillage libre, 154 « étoiles ».
    Ici : 1) inventaire (côtés par les 4 coins réels, _rect_sides) et comptes souhaités ;
    2) harmonisation par AJOUT de segments seulement jusqu'à ce que les côtés opposés de
    chaque face aient le même total et que les courbes bordant une face libre aient un
    nombre pair (full-quad) ; 3) une face impossible à harmoniser redevient libre (et elle
    seule) ; 4) application. Renvoie dict(bends, strips, patches, dropped), ou None si
    l'harmonisation n'aboutit pas (l'appelant revient à l'enchaînement face par face).
    seams (seam_chains) : coutures des faces-lanières retirées de la peau, planifiées comme une
    face à deux côtés (les deux chaînes : même total, nombres pairs) sans surface à mailler ;
    rendues dans out["seams"] (faces des coutures satisfaites)."""
    ref = set(pa.ref_faces)
    free = set(getattr(recipe, "free_faces", ()) or ())
    h = h0 * size_factor
    n_ref_faces: dict[int, int] = {}
    faces_of: dict[int, list[int]] = {}
    for g in pa.ref_faces:
        for c in pa.face_curves.get(g, []):
            n_ref_faces[abs(c)] = n_ref_faces.get(abs(c), 0) + 1
            faces_of.setdefault(abs(c), []).append(int(g))
    length: dict[int, float] = {}

    def clen(c):
        if c not in length:
            length[c] = gmsh.model.occ.getMass(1, c)
        return length[c]

    # ---- 1) inventaire ----
    plans = _bend_plans(pa, recipe, h0, size_factor, max_bend_angle_deg, min_bend_size_frac, angle_floor,
                        n_ref_faces)
    h_gen = min([h] + [6.0 * p["arc_len"] / p["n_arc"] for p in plans])
    for p in plans:                              # génératrices (paire 0-2), arcs (paire 1-3)
        p["target"] = [_even(int(round(max(sum(p["lens"][0]), sum(p["lens"][2])) / h_gen))), p["n_arc"]]
    planned = {p["face"] for p in plans}
    if strips_on:
        for f in sorted(ref - planned - free):
            try:
                r = _rect_sides(f, 0.5 * h0)
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                # repli : ancienne détection (2 plus longues courbes = grands côtés) pour une
                # lanière à bouts arrondis tangents (2 vrais coins seulement, part_015 face 12)
                p_old = _strip_plan_legacy(f, h, n_ref_faces, max_width_factor, min_length_ratio, min_size_frac)
                if p_old is not None:
                    plans.append(p_old)
                    planned.add(int(f))
                continue
            sides, lens, corners, _ = r
            tot = [sum(x) for x in lens]
            lp = 0 if tot[0] + tot[2] >= tot[1] + tot[3] else 1
            Lm = 0.5 * (tot[lp] + tot[lp + 2])
            if max(tot[lp], tot[lp + 2]) > 1.5 * min(tot[lp], tot[lp + 2]):
                continue
            w = gmsh.model.occ.getMass(2, f) / max(Lm, 1e-9)
            if w > max_width_factor * h or Lm < min_length_ratio * w:
                continue
            ok = True
            for e in (1 - lp, 3 - lp):
                # bout droit (éventuellement en biais) : corde ~ longueur ; pas trop oblique
                pa_ = np.array(gmsh.model.getValue(0, corners[e], []))
                pb_ = np.array(gmsh.model.getValue(0, corners[(e + 1) % 4], []))
                if np.linalg.norm(pb_ - pa_) < 0.9 * tot[e] or tot[e] > 3.0 * w:
                    ok = False
                # bout découpé ET partagé avec une autre face de peau : parité impossible à tenir
                if len(sides[e]) > 1 and any(n_ref_faces.get(c, 0) > 1 for c in sides[e]):
                    ok = False
                # micro-courbe dans un bout (chanfrein de 1,9 mm) : son sommet ferait une rangée
                # de 1,9 mm sur toute la longueur de la lanière -> laissée libre
                if len(sides[e]) > 1 and min(lens[e]) < min_size_frac * h:
                    ok = False
            if not ok:
                continue
            n_w = max(1, int(round(w / h)), len(sides[1 - lp]), len(sides[3 - lp]))
            if w / n_w < min_size_frac * h:
                continue
            tgt = [0, 0]
            tgt[lp] = max(1, int(round(Lm / h)))
            tgt[1 - lp] = n_w
            plans.append(dict(face=int(f), kind="lanière", sides=sides, lens=lens, corners=corners, target=tgt))
            planned.add(int(f))
    if patch_max_side is not None:
        for f in sorted(ref - planned - free):
            try:
                r = _rect_sides(f, 0.5 * h0)
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                p_old = _patch_plan_legacy(f, h, min(patch_max_side, 3.0 * h0))
                if p_old is not None:
                    plans.append(p_old)
                    planned.add(int(f))
                continue
            sides, lens, corners, turns = r
            if min(turns) < 45.0 or max(turns) > 135.0:
                continue
            tot = [sum(x) for x in lens]
            if max(tot) > patch_max_side:
                continue
            if max(tot[0], tot[2]) > 1.5 * min(tot[0], tot[2]) or max(tot[1], tot[3]) > 1.5 * min(tot[1], tot[3]):
                continue
            tgt = [_even(int(round(max(tot[k], tot[k + 2]) / h))) for k in (0, 1)]
            seg = [max(tot[k], tot[k + 2]) / tgt[k] for k in (0, 1)]
            lg, sh = (0, 1) if seg[0] >= seg[1] else (1, 0)
            if seg[lg] > 6.0 * seg[sh]:
                tgt[lg] = _even(int(np.ceil(max(tot[lg], tot[lg + 2]) / (6.0 * seg[sh]))))
            plans.append(dict(face=int(f), kind="patch", sides=sides, lens=lens, corners=corners, target=tgt))
            planned.add(int(f))

    # coutures : pseudo-faces (numéros < 0) à deux côtés opposés, sans surface
    seam_curves: set[int] = set()
    seam_of: dict[int, int] = {}
    for i, sm in enumerate(seams or []):
        ca, cb = [c for c, _, _ in sm["a"]], [c for c, _, _ in sm["b"]]
        la, lb = [clen(c) for c in ca], [clen(c) for c in cb]
        key = -1_000_000 - i
        seam_of[key] = sm["face"]
        seam_curves.update(ca + cb)
        plans.append(dict(face=key, kind="couture", sides=[ca, [], cb, []], lens=[la, [], lb, []], corners=None,
                          target=[_even(int(round(max(sum(la), sum(lb)) / h))), 0]))

    # ---- 2) harmonisation (ajouts seulement) ----
    dropped: list[int] = []
    solved = False
    for _round in range(10):
        active = {p["face"] for p in plans}
        even_req = {c for c, fs in faces_of.items() if any(g not in active for g in fs) and len(fs) > 1}
        even_req |= seam_curves
        cur: dict[int, int] = {}
        for p in plans:
            for k in (0, 1):
                for sd in (k, k + 2):
                    cnt = _split_count(p["lens"][sd], p["target"][k], [c in even_req for c in p["sides"][sd]])
                    for c, n in zip(p["sides"][sd], cnt):
                        cur[c] = max(cur.get(c, 1), n)
        for c, n1 in fixed.items():                 # déjà imposées ailleurs : constantes
            cur[c] = n1 - 1
        locked = set(fixed)
        converged = False
        for _ in range(max_sweeps):
            changed = False
            for p in plans:
                for a, b in ((0, 2), (1, 3)):
                    sa = sum(cur[c] for c in p["sides"][a])
                    sb = sum(cur[c] for c in p["sides"][b])
                    if sa == sb:
                        continue
                    small = p["sides"][a] if sa < sb else p["sides"][b]
                    cand = [c for c in small if c not in locked]
                    if not cand:
                        continue
                    for _k in range(abs(sa - sb)):
                        c = max(cand, key=lambda x: clen(x) / (cur[x] + 1))
                        cur[c] += 1
                    changed = True
            for c in even_req:
                if c in cur and cur[c] % 2 and c not in locked:
                    cur[c] += 1
                    changed = True
            if not changed:
                converged = True
                break
        bad = [p["face"] for p in plans
               if any(sum(cur[c] for c in p["sides"][a]) != sum(cur[c] for c in p["sides"][b])
                      for a, b in ((0, 2), (1, 3)))
               or any(c in even_req and cur[c] % 2 for sd in p["sides"] for c in sd)]
        if not bad:
            solved = True
            break
        dropped += bad
        plans = [p for p in plans if p["face"] not in bad]
    if not solved:
        return None                  # pas de solution : ancien enchaînement face par face

    # ---- 3) application ----
    out = dict(bends=[], strips=[], patches=[], seams=[], dropped=sorted(f for f in set(dropped) if f >= 0))
    key = dict(pli="bends", lanière="strips", patch="patches")
    for p in plans:
        want = {c: cur[c] + 1 for sd in p["sides"] for c in sd}
        try:
            for c, n1 in want.items():
                if c not in fixed:
                    gmsh.model.mesh.setTransfiniteCurve(c, n1)
            if p["kind"] == "couture":
                fixed.update(want)
                out["seams"].append(seam_of[p["face"]])
                continue
            if p["corners"] is None:
                gmsh.model.mesh.setTransfiniteSurface(p["face"])
            else:
                gmsh.model.mesh.setTransfiniteSurface(p["face"], cornerTags=p["corners"])
            gmsh.model.mesh.setRecombine(2, p["face"])
        except Exception:  # noqa: BLE001
            continue
        fixed.update(want)
        out[key[p["kind"]]].append(p["face"])
    return out


def even_boundaries(pa, structured: set[int], skip_curves: set[int] = frozenset()) -> int:
    """Recombinaison full-quad : gmsh divise par deux le maillage 1D de chaque courbe d'une
    face libre ; une courbe à nombre IMPAIR de segments (typiquement une arête CAD courte à
    1 segment) fait échouer la face (« 1D mesh cannot be divided by 2 »). Maille en 1D puis
    ajoute un segment à ces courbes (hors transfinis imposés ; composites comprises : gmsh
    y garde le nœud commun, la parité porte sur chaque constituant). Remplace
    l'ancien raffinement autour des arêtes courtes, qui ne servait qu'à ça et produisait
    des anneaux d'éléments minuscules. Renvoie le nombre de courbes corrigées."""
    free_faces = [f for f in pa.ref_faces if f not in structured]
    struct_curves = {abs(c) for f in structured for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
    curves = {abs(c) for f in free_faces for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
    gmsh.model.mesh.generate(1)
    n_fix = 0
    for c in sorted(curves - struct_curves - set(skip_curves)):
        try:
            n = sum(len(t) for t in gmsh.model.mesh.getElements(1, c)[1])
        except Exception:  # noqa: BLE001
            continue
        if n % 2:
            gmsh.model.mesh.setTransfiniteCurve(c, n + 2)
            n_fix += 1
    if n_fix:
        gmsh.model.mesh.clear()
    return n_fix


def apply_strategy(pa, recipe, h0: float, patches_on: bool = True,
                   max_bend_angle_deg: float = 15.0, min_bend_size_frac: float = 0.0,
                   bend_angle_floor: bool = False, contour_arcs_on: bool = False,
                   large_face_elements: float = 0.0, large_face_alg: int = 6, harmonize: bool = False,
                   patch_max_frac: float = 8.0, seams: list | None = None) -> dict:
    s = recipe.strategy
    seams = seams or []
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
    info = dict(strategy=s, compounds=0)
    sf = 2.0 if s == "subdiv" else 1.0
    fixed: dict[int, int] = {}
    structured = getattr(recipe, "structured", True)
    plan = None
    if harmonize and structured and s in ("conform", "compound", "blossom", "subdiv"):
        # plis, lanières et petites faces planifiés ensemble, comptes harmonisés
        plan = structured_plan(pa, recipe, h0, sf, fixed, max_bend_angle_deg, min_bend_size_frac,
                               bend_angle_floor, strips_on=getattr(recipe, "strips", True),
                               patch_max_side=(float("inf") if patches_on else patch_max_frac * h0)
                               if s in ("conform", "compound", "blossom") else None, seams=seams)
    info["structured_harmonized"] = plan is not None
    # coutures hors planification (plis libres, pas de solution, couture écartée) : comptes
    # imposés directement, AVANT les faces structurées de l'ancien enchaînement
    seam_done = list(plan["seams"]) if plan is not None else []
    if s != "qqs":
        seam_done += seam_counts([sm for sm in seams if sm["face"] not in seam_done], h0 * sf, fixed)
    info["seams"] = len(seam_done)
    info["seam_curves"] = sorted({c for sm in seams for c, _, _ in sm["a"] + sm["b"]})
    if plan is not None:
        info["structured_bends"] = len(plan["bends"])
        info["structured_strips"] = plan["strips"]
        info["structured_list"] = plan["bends"] + plan["strips"]
        info["structured_patches"] = len(plan["patches"])
        info["structured_patch_list"] = plan["patches"]
        info["structured_dropped"] = plan["dropped"]
        bends = plan["bends"] + plan["strips"] + plan["patches"]
    else:
        bends = structured_bends(pa, recipe, h0, sf, fixed, max_bend_angle_deg, min_bend_size_frac,
                                 bend_angle_floor) \
            if structured and s in ("conform", "compound", "blossom", "subdiv") else []
        info["structured_bends"] = len(bends)
        strips = []
        if structured and s in ("conform", "compound", "blossom", "subdiv") and getattr(recipe, "strips", True):
            strips = structured_strips(pa, recipe, h0, fixed, set(bends), sf)
        info["structured_strips"] = strips
        bends = bends + strips
        info["structured_list"] = list(bends)
        if structured and s in ("conform", "compound", "blossom"):
            # toutes les faces à 4 côtés si demandé, sinon seulement les petites
            patches = structured_patches(pa, h0, fixed, set(bends), sf,
                                         max_side=float("inf") if patches_on else 3.0 * h0)
            info["structured_patches"] = len(patches)
            info["structured_patch_list"] = list(patches)
            bends = bends + patches
    merged_c: set[int] = set()
    if s != "qqs":
        fixed_c = {abs(c) for f in bends for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
        fixed_c |= set(info["seam_curves"])
        # points/arêtes CAD trop proches (rayon isolé, jonction serrée...) : fusionnés en
        # courbe composite avant maillage, sinon un point CAD isolé impose un nœud et un
        # éventail d'éléments minuscules autour, même si le champ de taille est grossier
        merge_frac = min_bend_size_frac if min_bend_size_frac > 0 else 0.1
        merge_len = min(merge_frac * h0, 0.5 * pa.classification["t_median"])
        merged = merge_micro_curves(merge_len, fixed_c)
        info["micro_curve_merges"] = len(merged)
        info["merged_curves"] = sorted(c for g in merged for c in g)
        merged_c = set(info["merged_curves"])
    # arcs du contour APRÈS la fusion : un arc fusionné avec une micro-courbe voisine reste
    # libre (l'imposer empêchait la fusion -> rosette autour de la micro-courbe, part_016)
    arcs = []
    if structured and s in ("conform", "compound", "blossom") and contour_arcs_on:
        arcs = contour_arcs(pa, h0, fixed, max_bend_angle_deg, min_size_frac=max(min_bend_size_frac, 0.1),
                            even=(s != "blossom"), skip=merged_c)
    info["contour_arcs"] = len(arcs)
    info["contour_arc_list"] = arcs
    periodic = [f for f in pa.ref_faces if _is_periodic(f) and f not in bends]
    if s in ("conform", "blossom"):
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3 if s == "conform" else 1)
        for f in periodic:
            gmsh.model.mesh.setCompound(2, [f])
        info["compounds"] = len(periodic)
    elif s == "compound":
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
        n = 0
        bset = set(bends)
        for g in tangent_groups(pa, recipe.tangent_angle_deg, exclude=bset):
            if len(g) > 1 or _is_periodic(g[0]):
                gmsh.model.mesh.setCompound(2, g)
                n += 1
        info["compounds"] = n
    elif s == "subdiv":
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        # la subdivision divise la taille par 2 : on compense
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 2.0)
    elif s == "qqs":
        gmsh.option.setNumber("Mesh.Algorithm", 11)
    else:
        raise ValueError(s)
    # fusions locales adaptatives : face étroite + voisine tangente en surface composite
    groups: dict[int, set[int]] = {}
    skip = set(bends)
    for a, b in getattr(recipe, "merge", ()) or ():
        a, b = int(a), int(b)
        if a in skip or b in skip or a not in pa.ref_faces or b not in pa.ref_faces:
            continue
        ga, gb = groups.get(a, {a}), groups.get(b, {b})
        g = ga | gb
        for f in g:
            groups[f] = g
    done_groups = []
    for g in {id(g): g for g in groups.values()}.values():
        try:
            gmsh.model.mesh.setCompound(2, sorted(g))
            done_groups.append(sorted(g))
        except Exception:  # noqa: BLE001
            pass
    info["merged_groups"] = done_groups
    # remaillage local adaptatif : algorithme 2D imposé face par face
    fa = [(int(f), int(a)) for f, a in getattr(recipe, "face_alg", ()) or ()]
    ref = set(pa.ref_faces)
    # très grandes faces (coque upper part_011 : 2 BSpline de 32 m²) : l'algorithme 8 travaille
    # dans l'espace paramétrique et n'y tient pas la taille (26 000 quads énormes et étirés au
    # lieu de 790 000) ; l'algorithme 6 (Frontal-Delaunay) la tient -> choisi d'emblée
    big = []
    if large_face_elements > 0 and s in ("conform", "blossom", "compound"):
        done_fa = {f for f, _ in fa}
        for f in ref - done_fa - set(bends):
            try:
                if gmsh.model.occ.getMass(2, f) / max(h0 * h0, 1e-9) > large_face_elements:
                    gmsh.model.mesh.setAlgorithm(2, f, int(large_face_alg))
                    big.append(f)
            except Exception:  # noqa: BLE001
                pass
    info["large_faces"] = big
    for f, a in fa:
        if f in ref:
            gmsh.model.mesh.setAlgorithm(2, f, a)
    info["face_alg"] = len(fa)
    return info


@dataclass
class QuadMesh:
    X: np.ndarray            # (N,3)
    node_tags: np.ndarray    # (N,) tags gmsh
    quads: np.ndarray        # (Q,4) indices locaux
    tris: np.ndarray         # (R,3)
    quad_face: np.ndarray    # (Q,) face CAD
    tri_face: np.ndarray     # (R,)
    other_elems: int = 0
    fixed: np.ndarray | None = None   # (N,) nœud sur une courbe / un sommet CAD

    @property
    def n_elems(self):
        return len(self.quads) + len(self.tris)


def extract_reference_mesh(pa) -> QuadMesh:
    quads, tris, qf, tf, other = [], [], [], [], 0
    for f in pa.ref_faces:
        try:
            ets, _, ens = gmsh.model.mesh.getElements(2, f)
        except Exception:  # noqa: BLE001
            continue
        for et, en in zip(ets, ens):
            if et == 3:
                q = en.reshape(-1, 4)
                quads.append(q)
                qf.append(np.full(len(q), f))
            elif et == 2:
                t = en.reshape(-1, 3)
                tris.append(t)
                tf.append(np.full(len(t), f))
            else:
                other += len(en)
    Q = np.concatenate(quads).astype(np.int64) if quads else np.zeros((0, 4), np.int64)
    R = np.concatenate(tris).astype(np.int64) if tris else np.zeros((0, 3), np.int64)
    used = np.unique(np.concatenate([Q.ravel(), R.ravel()]))
    tags, coords, _ = gmsh.model.mesh.getNodes()
    pos = np.full(int(tags.max()) + 1, -1, np.int64)
    pos[tags.astype(np.int64)] = np.arange(len(tags))
    X = coords.reshape(-1, 3)[pos[used]]
    loc = np.full(int(tags.max()) + 1, -1, np.int64)
    loc[used] = np.arange(len(used))
    interior = set()
    for f in pa.ref_faces:
        try:
            it, _, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=False)
            interior.update(int(t) for t in it)
        except Exception:  # noqa: BLE001
            pass
    fixed = np.array([int(t) not in interior for t in used], dtype=bool)
    return QuadMesh(X=X, node_tags=used, quads=loc[Q], tris=loc[R],
                    quad_face=np.concatenate(qf) if qf else np.zeros(0, np.int64),
                    tri_face=np.concatenate(tf) if tf else np.zeros(0, np.int64), other_elems=other,
                    fixed=fixed)


def _chain_node_tags(chain) -> list[int]:
    """Numéros gmsh des nœuds d'une chaîne (seam_chains), dans l'ordre de parcours."""
    seq: list[int] = []
    for c, p0, p1 in chain:
        tags, _, par = gmsh.model.mesh.getNodes(1, c, includeBoundary=False, returnParametricCoord=True)
        order = np.argsort(np.asarray(par, float))
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        x0 = np.array(gmsh.model.getValue(0, p0, []), float)
        if np.linalg.norm(np.array(gmsh.model.getValue(1, c, [lo[0]]), float) - x0) > \
                np.linalg.norm(np.array(gmsh.model.getValue(1, c, [hi[0]]), float) - x0):
            order = order[::-1]
        t0 = int(gmsh.model.mesh.getNodes(0, p0)[0][0])
        if not seq or seq[-1] != t0:
            seq.append(t0)
        seq += [int(tags[i]) for i in order]
        seq.append(int(gmsh.model.mesh.getNodes(0, p1)[0][0]))
    return seq


def _collapse_chords(Q: np.ndarray, R: np.ndarray, X: np.ndarray, pinned: np.ndarray, max_merges: int = 5000):
    """Colonnes de quads refermées après une soudure : un quad dont une arête s'est réduite à un
    point (le bout d'une face-lanière posé sur le bord d'une face voisine non coupée, âme des
    cadres upper 002 / 004) est retiré et son arête OPPOSÉE est refermée à son tour, de proche en
    proche jusqu'à un bord (« chord collapse ») : la colonne en coin disparaît, le maillage reste
    100 % quads et sans trou. Nœud fusionné au milieu des deux, sauf si l'un est épinglé (sommet
    CAO, nœud d'une couture). Avant cela, deux quads réduits à deux triangles ACCOLÉS autour du
    même nœud (face libre : le bout de lanière y porte 2 segments, parité) sont réunis en un
    seul quad, sans propagation. Renvoie (Q, R, quads gardés, triangles gardés, nombre de
    fusions), ou None si la propagation ne s'arrête pas. X modifié en place."""
    Q, R = Q.copy(), R.copy()
    keep_q, keep_r = np.ones(len(Q), bool), np.ones(len(R), bool)
    n_merge = 0

    def tri(i):
        """Quad i réduit à un triangle (p, c, d) dans son sens de parcours, ou None."""
        q = Q[i].tolist()
        if len(set(q)) != 3:
            return None
        k = next(k for k in range(4) if q[k] == q[(k + 1) % 4])
        return q[k], q[(k + 2) % 4], q[(k + 3) % 4]

    # triangles accolés autour du même nœud -> un quad
    deg = [int(i) for i in np.nonzero((Q == np.roll(Q, -1, axis=1)).any(axis=1))[0]]
    tris = {i: tri(i) for i in deg}
    for i in deg:
        if not keep_q[i] or tris[i] is None:
            continue
        p, c, d = tris[i]
        for j in deg:
            if j == i or not keep_q[j] or tris[j] is None or tris[j][0] != p:
                continue
            _, c2, d2 = tris[j]
            if d2 == c and c2 != d:
                Q[i] = (p, c2, c, d)
            elif c2 == d and d2 != c:
                Q[i] = (p, c, d, d2)
            else:
                continue
            keep_q[j] = False
            tris[i] = None
            break
    while True:
        deg = np.nonzero(keep_q & ((Q == np.roll(Q, -1, axis=1)).any(axis=1)))[0]
        if not len(deg):
            break
        i = int(deg[0])
        q = Q[i].tolist()
        keep_q[i] = False
        if len(set(q)) != 3:
            continue                                   # réduit à une arête ou à un point
        k = next(k for k in range(4) if q[k] == q[(k + 1) % 4])
        c, d = q[(k + 2) % 4], q[(k + 3) % 4]
        if pinned[c] and pinned[d]:
            return None                                # deux nœuds imposés : pas de fermeture possible
        n_merge += 1
        if n_merge > max_merges:
            return None
        if pinned[d]:
            X[c] = X[d]
        elif not pinned[c]:
            X[c] = 0.5 * (X[c] + X[d])
        pinned[c] = pinned[c] or pinned[d]
        Q[Q == d] = c
        R[R == d] = c
    if len(R):
        keep_r = np.array([len(set(t)) == 3 for t in R.tolist()], bool)
    return Q, R, keep_q, keep_r, n_merge


def weld_seams(qm: QuadMesh, seams: list[dict], max_shift_frac: float = 0.45) -> dict:
    """Soude les deux bords de chaque couture (faces-lanières retirées de la peau, seam_chains) :
    le k-ième nœud de la chaîne b est remplacé par le k-ième de la chaîne a (qui garde sa
    position, sur sa courbe CAO). Les deux chaînes ont le même nombre de nœuds (apply_strategy).
    Une couture est refusée, et laissée ouverte (fissure vue par le critère de continuité), si
    les comptes diffèrent ou si un nœud devrait glisser de plus de max_shift_frac x le pas local
    le long du bord. Les quads réduits à un triangle par la soudure (bout de lanière sur le bord
    d'une face voisine) sont résorbés avec leur colonne (_collapse_chords) ; si c'est
    impossible, rien n'est soudé. Modifie qm en place.
    Renvoie dict(seams, nodes, failed, max_shift_mm, chord_merges)."""
    info = dict(seams=0, nodes=0, failed=[], max_shift_mm=0.0, chord_merges=0)
    if not seams:
        return info
    lk = {int(t): i for i, t in enumerate(qm.node_tags)}
    remap = np.arange(len(qm.X))
    for sm in seams:
        try:
            ta, tb = _chain_node_tags(sm["a"]), _chain_node_tags(sm["b"])
            if len(ta) != len(tb) or any(t not in lk for t in ta + tb):
                info["failed"].append(dict(face=sm["face"], reason=f"{len(ta)} / {len(tb)} nœuds"))
                continue
            ia, ib = np.array([lk[t] for t in ta]), np.array([lk[t] for t in tb])
            d = np.linalg.norm(qm.X[ia] - qm.X[ib], axis=1)
            seg = np.linalg.norm(np.diff(qm.X[ia], axis=0), axis=1)
            step = np.minimum(np.r_[seg[0], seg], np.r_[seg, seg[-1]])
            if (d > max_shift_frac * step).any():
                k = int(np.argmax(d / np.maximum(step, 1e-30)))
                info["failed"].append(dict(face=sm["face"], reason=f"écart {d[k]:.3f} mm pour un pas de {step[k]:.3f}"))
                continue
            remap[ib] = ia
            # bouts de la lanière restés dans le modèle (bord d'une face voisine non coupée) :
            # tous leurs nœuds vont sur le nœud soudé du bout
            for cs, tgt in zip(sm.get("links") or ((), ()), (ia[0], ia[-1])):
                for c in cs:
                    for t in gmsh.model.mesh.getNodes(1, c, includeBoundary=True)[0]:
                        if int(t) in lk:
                            remap[lk[int(t)]] = tgt
            info["seams"] += 1
            info["nodes"] += int((ia != ib).sum())
            info["max_shift_mm"] = max(info["max_shift_mm"], float(d.max()))
        except Exception as e:  # noqa: BLE001
            info["failed"].append(dict(face=sm.get("face"), reason=f"{type(e).__name__}: {e}"[:120]))
    if not info["nodes"]:
        return info
    for _ in range(8):                      # lanières accolées : soudures en chaîne
        nxt = remap[remap]
        if (nxt == remap).all():
            break
        remap = nxt
    # nœuds épinglés : sommets CAO et nœuds des coutures (ils restent sur leur courbe)
    pinned = np.zeros(len(qm.X), bool)
    for _, p_ in gmsh.model.getEntities(0):
        try:
            pinned[[lk[int(t)] for t in gmsh.model.mesh.getNodes(0, p_)[0] if int(t) in lk]] = True
        except Exception:  # noqa: BLE001
            pass
    pinned[remap[np.nonzero(remap != np.arange(len(remap)))[0]]] = True
    X = qm.X.copy()
    r = _collapse_chords(remap[qm.quads], remap[qm.tris], X, pinned)
    if r is None:
        info.update(seams=0, nodes=0)
        info["failed"].append(dict(face=None, reason="colonne d'éléments impossible à refermer"))
        return info
    Q, R, okq, okr, info["chord_merges"] = r
    qm.X = X
    Q, R = Q[okq], R[okr]
    used = np.unique(np.concatenate([Q.ravel(), R.ravel()]))
    loc = np.full(len(qm.X), -1, np.int64)
    loc[used] = np.arange(len(used))
    qm.quads, qm.tris = loc[Q], loc[R]
    qm.quad_face, qm.tri_face = qm.quad_face[okq], qm.tri_face[okr]
    qm.X, qm.node_tags = qm.X[used], qm.node_tags[used]
    if qm.fixed is not None:
        qm.fixed = qm.fixed[used]
    return info


def nodes_on_curves(qm: QuadMesh, curves) -> np.ndarray:
    """Indices locaux des nœuds du maillage de référence situés sur les courbes données."""
    lookup = {int(t): i for i, t in enumerate(qm.node_tags)}
    out = set()
    for c in curves:
        try:
            tags, _, _ = gmsh.model.mesh.getNodes(1, c, includeBoundary=True)
        except Exception:  # noqa: BLE001
            continue
        out.update(lookup[int(t)] for t in tags if int(t) in lookup)
    return np.array(sorted(out), dtype=np.int64)
