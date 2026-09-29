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
    """Relit la géométrie préparée et ne garde que les faces de la peau de référence."""
    gmsh.model.occ.importShapes(pa.brep)
    gmsh.model.occ.synchronize()
    ref = set(pa.ref_faces)
    gmsh.model.removeEntities(gmsh.model.getEntities(3))
    others = [(2, f) for _, f in gmsh.model.getEntities(2) if f not in ref]
    if others:
        gmsh.model.removeEntities(others, recursive=True)


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


def _split_count(lengths: list[float], total: int) -> list[int]:
    """Répartit `total` segments sur une chaîne de courbes, au prorata des longueurs (>= 1 chacune)."""
    L = np.asarray(lengths, float)
    if len(L) == 1:
        return [total]
    raw = np.maximum(1, np.floor(total * L / L.sum()).astype(int))
    while raw.sum() > total and (raw > 1).any():
        raw[np.argmax(np.where(raw > 1, raw - total * L / L.sum(), -np.inf))] -= 1
    while raw.sum() < total:
        raw[np.argmax(total * L / L.sum() - raw)] += 1
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


def _bend_sides(f: int):
    """Contour d'une face de pli découpé en 4 côtés : (génératrice, arcs, génératrice, arcs).

    Les deux génératrices sont les deux plus longues courbes, non adjacentes ; les
    chaînes d'arcs sont ce qui reste entre elles. Retourne (côtés, coins) ou None.
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
    i1, i2 = sorted(sorted(range(n), key=lambda i: -L[i])[:2])
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
                r = _bend_sides(f)
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
        want = {}
        for side, ln, tot in ((sides[0], lens[0], n_gen), (sides[1], lens[1], n_arc),
                              (sides[2], lens[2], n_gen), (sides[3], lens[3], n_arc)):
            for c, k in zip(side, _split_count(ln, tot)):
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
                tot = tot_fixed[0] if tot_fixed else _even(int(round(max(L[a_], L[b_]) / (h0 * size_factor))))
                for sd in (a_, b_):
                    if all(c in fixed for c in sides[sd]):
                        continue
                    for c, k in zip(sides[sd], _split_count(slens[sd], tot)):
                        want[c] = k + 1
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
                   bend_angle_floor: bool = False) -> dict:
    s = recipe.strategy
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
    info = dict(strategy=s, compounds=0)
    sf = 2.0 if s == "subdiv" else 1.0
    fixed: dict[int, int] = {}
    structured = getattr(recipe, "structured", True)
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
    if s != "qqs":
        fixed_c = {abs(c) for f in bends for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
        # points/arêtes CAD trop proches (rayon isolé, jonction serrée...) : fusionnés en
        # courbe composite avant maillage, sinon un point CAD isolé impose un nœud et un
        # éventail d'éléments minuscules autour, même si le champ de taille est grossier
        merge_frac = min_bend_size_frac if min_bend_size_frac > 0 else 0.1
        merge_len = min(merge_frac * h0, 0.5 * pa.classification["t_median"])
        merged = merge_micro_curves(merge_len, fixed_c)
        info["micro_curve_merges"] = len(merged)
        info["merged_curves"] = sorted(c for g in merged for c in g)
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
