"""Réparation topologique locale du maillage quad : bascule d'arête + lissage local.

Deux quads adjacents (même face CAD) forment un hexagone A-B-U-C-D-V ; il existe
trois façons de le découper en deux quads (diagonales U-V, B-D, A-C). Pour chaque
alternative, les nœuds libres (intérieurs aux faces, pas sur une courbe CAD) du
voisinage sont relissés (Laplacien), puis on garde la configuration de meilleure
qualité minimale. Aucun nœud ajouté, 100 % quads conservé.

Cas typique : quad dont trois nœuds sont sur un arc de bord concave (angle ~175° au
nœud central) ; seule la diagonale issue de ce nœud coupe l'angle, et elle n'est
bonne qu'après repositionnement des nœuds voisins.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .quad import QuadMesh


def quad_quality(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jacobien normalisé 2D minimal aux 4 coins (normale = produit des diagonales)."""
    Q = np.asarray(Q, dtype=np.int64).reshape(-1, 4)
    P = X[Q]
    n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-300)
    out = np.full(len(Q), np.inf)
    for k in range(4):
        e1 = P[:, (k + 1) % 4] - P[:, k]
        e2 = P[:, (k - 1) % 4] - P[:, k]
        s = np.einsum("ij,ij->i", np.cross(e1, e2), n)
        s /= np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-300)
        out = np.minimum(out, s)
    return out


def _quad_shape(X: np.ndarray, Q: np.ndarray):
    """Élancement (arête max / min), angle interne min et max (deg) par quad.

    Sert à cibler la réparation sur les mêmes critères de forme que le verdict de
    qualité (angle max, angle min, élancement), pas seulement sur le jacobien.
    """
    Q = np.asarray(Q, dtype=np.int64).reshape(-1, 4)
    if len(Q) == 0:
        z = np.zeros(0)
        return z, z, z
    P = [X[Q[:, k]] for k in range(4)]
    edges = [P[(k + 1) % 4] - P[k] for k in range(4)]
    lens = np.stack([np.linalg.norm(e, axis=1) for e in edges], 1)
    aspect = lens.max(1) / np.maximum(lens.min(1), 1e-300)
    angs = []
    for k in range(4):
        u = -edges[(k - 1) % 4]
        v = edges[k]
        c = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    angs = np.stack(angs, 1)
    return aspect, angs.min(1), angs.max(1)


def _rot_to_edge(q, u, v):
    """Rotation cyclique de q pour que l'arête u->v soit en positions 2->3."""
    for r in range(4):
        qq = q[r:] + q[:r]
        if qq[2] == u and qq[3] == v:
            return qq
    return None


class _Local:
    """Évalue une modification locale (quads remplacés + lissage des nœuds libres)."""

    def __init__(self, X, Q, fixed, node_quads):
        self.X, self.Q, self.fixed, self.node_quads = X, Q, fixed, node_quads

    def evaluate(self, repl: dict[int, list[int]], iters: int = 20, rings: int = 2):
        # quads concernés : remplacés + voisins par nœud sur `rings` anneaux
        ring = set(repl)
        nodes = {n for q in repl.values() for n in q}
        for _ in range(rings):
            for n in nodes:
                ring.update(self.node_quads[n])
            nodes = {n for i in ring for n in repl.get(i, self.Q[i])}
        quads = {i: repl.get(i, self.Q[i]) for i in ring}
        free = [n for n in {n for q in quads.values() for n in q} if not self.fixed[n]]
        # voisinage des nœuds libres (arêtes des quads du voisinage + quads hors voisinage)
        nbrs = defaultdict(set)
        all_q = dict(quads)
        for n in free:
            for i in self.node_quads[n]:
                all_q.setdefault(i, repl.get(i, self.Q[i]))
        for q in all_q.values():
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                nbrs[a].add(b)
                nbrs[b].add(a)
        Xl = {n: self.X[n].copy() for n in {m for q in all_q.values() for m in q}}
        for _ in range(iters):
            for n in free:
                if nbrs[n]:
                    Xl[n] = np.mean([Xl[m] for m in nbrs[n]], axis=0)
        idx = {n: k for k, n in enumerate(Xl)}
        XX = np.array(list(Xl.values()))
        ids = list(all_q)
        qa = np.array([[idx[n] for n in all_q[i]] for i in ids])
        qq = quad_quality(XX, qa)
        self.last = dict(zip(ids, qq.tolist()))
        # score : qualité des quads remplacés, plafonnée par la plus mauvaise des voisins
        # (les voisins ne doivent pas descendre sous leur état initial)
        rep = min(self.last[i] for i in repl)
        oth = [i for i in ids if i not in repl]
        if oth:
            before = quad_quality(self.X, np.array([self.Q[i] for i in oth], dtype=np.int64))
            after = np.array([self.last[i] for i in oth])
            # un voisin ne pénalise que s'il se dégrade
            worse = after < before - 1e-9
            if worse.any():
                rep = min(rep, float(after[worse].min()))
        return float(rep), Xl, ids


def flip_repair(qm: QuadMesh, fixed: np.ndarray, threshold: float = 0.35, passes: int = 4,
                gain: float = 0.05, shape: dict | None = None, time_budget_s: float = 120.0,
                frozen_faces=frozenset(), hard_flip: float = 0.1) -> int:
    """Bascule + lissage local des quads médiocres.

    `threshold` : jacobien 2D en dessous duquel un quad est visé. `shape` (optionnel,
    seuils `max_angle_deg` / `min_angle_deg` / `max_aspect_ratio`) vise en plus les
    quads qui violent les critères de forme cibles même si leur jacobien reste au-dessus
    du seuil : un quad à 167° a un jacobien ~0.22 > 0.2 et échappait sinon à la réparation.
    frozen_faces : faces transfinies, jamais basculées (une bascule y crée deux nœuds à 3 et
    5 quads dans des rangées régulières, upper part_001) : lissage seul (topologie intacte ;
    il déplie les quads retournés au bout en biais d'une patte).
    """
    import time
    t_start = time.time()
    X = qm.X.copy()
    Q = [list(map(int, q)) for q in qm.quads]
    F = qm.quad_face.tolist()
    n_flips = 0
    for _ in range(passes):
        if time.time() - t_start > time_budget_s:
            break
        qual = quad_quality(X, np.array(Q, dtype=np.int64))
        target = qual < threshold
        if shape:
            aspect, amin, amax = _quad_shape(X, np.array(Q, dtype=np.int64))
            target |= ((amax > shape["max_angle_deg"]) | (amin < shape["min_angle_deg"])
                       | (aspect > shape["max_aspect_ratio"]))
        edge_q = defaultdict(list)
        node_quads = defaultdict(list)
        for i, q in enumerate(Q):
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                edge_q[(min(a, b), max(a, b))].append(i)
                node_quads[q[k]].append(i)
        loc = _Local(X, Q, fixed, node_quads)
        touched = set()
        changed = 0
        for i in np.argsort(qual):
            if not target[i]:
                continue
            if i in touched:
                continue
            # très grands maillages (10^5 quads et plus) : réparation bornée dans le temps
            # (pires quads d'abord)
            if time.time() - t_start > time_budget_s:
                break
            q1 = Q[i]
            cur_q = float(qual[i])
            best = None
            # option 0 : simple lissage local, sans changement de topologie
            qs, Xs, _ = loc.evaluate({i: q1})
            if qs > cur_q + gain:
                best = (qs, {i: q1}, Xs)
            # face transfinie : bascule seulement pour un quad de jacobien médiocre (< seuil de
            # réparation), pas pour un défaut de forme seul (élancement, angle) : une étoile vaut
            # mieux qu'un hexa retourné après décalage (part_015 : pli R3, quad à 0,1-0,2)
            for k in range(4 if (F[i] not in frozen_faces or cur_q < max(hard_flip, threshold)) else 0):
                u, v = q1[k], q1[(k + 1) % 4]
                nb = [j for j in edge_q[(min(u, v), max(u, v))] if j != i]
                if len(nb) != 1:
                    continue
                j = nb[0]
                if j in touched or F[j] != F[i]:
                    continue
                qa = _rot_to_edge(q1, u, v)          # (A, B, U, V)
                qb = _rot_to_edge(Q[j], v, u)        # (C, D, V, U)
                if qa is None or qb is None:
                    continue
                A, B, U, V = qa
                C, D = qb[0], qb[1]
                for alt in (([B, U, C, D], [D, V, A, B]), ([A, B, U, C], [C, D, V, A])):
                    if len(set(alt[0])) < 4 or len(set(alt[1])) < 4:
                        continue
                    # bascule seule, puis bascule + lissage local : on garde la meilleure
                    for iters in (0, 20):
                        qq, Xs, _ = loc.evaluate({i: alt[0], j: alt[1]}, iters=iters)
                        if qq > cur_q + gain and (best is None or qq > best[0]):
                            best = (qq, {i: alt[0], j: alt[1]}, Xs)
            if best is not None:
                _, repl, Xs = best
                for idx_q, q in repl.items():
                    Q[idx_q] = q
                    touched.add(idx_q)
                for n, p in Xs.items():
                    if not fixed[n]:
                        X[n] = p
                # les quads voisins dont des nœuds ont bougé sont figés pour cette passe
                for n in Xs:
                    touched.update(node_quads[n])
                changed += 1
        n_flips += changed
        if not changed:
            break
    qm.quads = np.array(Q, dtype=np.int64).reshape(-1, 4)
    qm.X[:] = X
    return n_flips


def fix_micro_edges(qm: QuadMesh, eps: float, max_iter: int = 3) -> int:
    """Arêtes de quad plus courtes que `eps` (imposées par une micro-arête CAD) :
    l'une des extrémités glisse vers le voisin qui prolonge l'arête (même bord),
    à 40 % de la distance. Topologie inchangée ; accepté seulement si la qualité
    minimale des quads concernés s'améliore. Retourne le nombre de nœuds déplacés."""
    X = qm.X
    Q = qm.quads
    if len(Q) == 0:
        return 0
    moved = 0
    for _ in range(max_iter):
        e = np.concatenate([Q[:, [k, (k + 1) % 4]] for k in range(4)])
        e = np.unique(np.sort(e, axis=1), axis=0)
        L = np.linalg.norm(X[e[:, 1]] - X[e[:, 0]], axis=1)
        short = e[L < eps]
        if not len(short):
            break
        nbrs = defaultdict(set)
        for a, b in e:
            nbrs[int(a)].add(int(b))
            nbrs[int(b)].add(int(a))
        node_quads = defaultdict(list)
        for i, q in enumerate(Q):
            for n in q:
                node_quads[int(n)].append(i)
        changed = 0
        for a0, b0 in short:
            best = None
            for a, b in ((int(a0), int(b0)), (int(b0), int(a0))):
                u = X[a] - X[b]
                nu = np.linalg.norm(u)
                if nu == 0:
                    continue
                u = u / nu
                for c in nbrs[a] - {b}:
                    v = X[c] - X[a]
                    lv = np.linalg.norm(v)
                    if lv < 5 * eps:
                        continue
                    cosang = float(v @ u / lv)
                    if cosang > 0.7 and (best is None or cosang > best[0]):
                        best = (cosang, a, c)
            if best is None:
                continue
            _, a, c = best
            quads = node_quads[a]
            before = quad_quality(X, Q[quads]).min()
            old = X[a].copy()
            X[a] = old + 0.4 * (X[c] - old)
            after = quad_quality(X, Q[quads]).min()
            if after > before:
                changed += 1
            else:
                X[a] = old
        moved += changed
        if not changed:
            break
    return moved


def _boundary_loops(Q: np.ndarray):
    """Contours du maillage quad (arêtes à un seul quad) : listes ordonnées de nœuds,
    quad propriétaire de chaque arête, nœuds de degré != 2 (pincements) à part."""
    owner = {}
    count = defaultdict(int)
    for i, q in enumerate(Q):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            key = (min(a, b), max(a, b))
            count[key] += 1
            owner[key] = i
    bnd = [k for k, c in count.items() if c == 1]
    adj = defaultdict(list)
    for a, b in bnd:
        adj[a].append(b)
        adj[b].append(a)
    pinch = {n for n, v in adj.items() if len(v) != 2}
    seen, loops = set(), []
    for s in adj:
        if s in seen or s in pinch:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        closed = False
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == s:
                closed = True
                break
            if cur in seen or cur in pinch:
                break
            seen.add(cur)
            loop.append(cur)
        if closed and len(loop) >= 4:
            loops.append(loop)
    return loops, owner, pinch


def relax_boundary(qm: QuadMesh, protect: np.ndarray, h: float, short_frac: float = 0.35,
                   corner_deg: float = 25.0, window: int = 3, iters: int = 30,
                   shape: dict | None = None, log: list | None = None) -> int:
    """Arêtes courtes du CONTOUR de la peau (bord libre) : redistribution des nœuds le long
    du contour + lissage des nœuds intérieurs voisins.

    Une arête CAD de ~1 mm sur un contour lisse (facette de chant, raccord de deux chants)
    impose 2 segments de 0,5 mm (parité du full-quad) : quads d'élancement 10-17 à côté
    d'éléments de h0, hors cibles -> la recette nominale est rejetée au profit d'une recette
    3-4 fois plus fine (part_016, 017, 012). Le sommet CAD n'a pas de sens mécanique sur un
    contour lisse : ses nœuds glissent le long du contour (polyligne d'origine), à pas
    uniforme entre deux ancrages (coins réels > corner_deg, nœuds protégés : plis et faces
    transfinis, trous ; au plus `window` nœuds de part et d'autre). Accepté seulement si
    l'élancement max et le jacobien min des quads touchés s'améliorent (ou restent bons).
    Renvoie le nombre de fenêtres relaxées."""
    X, Q = qm.X, qm.quads
    if len(Q) == 0 or h <= 0:
        return 0
    loops, owner, pinch = _boundary_loops(Q)
    node_quads = defaultdict(list)
    for i, q in enumerate(Q):
        for n in q:
            node_quads[int(n)].append(i)
    nbrs = defaultdict(set)
    for q in Q:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            nbrs[a].add(b)
            nbrs[b].add(a)
    fixed = qm.fixed
    max_ar = (shape or {}).get("max_aspect_ratio", 10.0)
    n_done = 0

    for loop in loops:
        m = len(loop)
        L = np.array(loop)
        P = X[L]
        e_out = np.roll(P, -1, 0) - P
        e_in = P - np.roll(P, 1, 0)
        seg = np.linalg.norm(e_out, axis=1)
        c = np.einsum("ij,ij->i", e_in, e_out) / np.maximum(
            np.linalg.norm(e_in, axis=1) * seg, 1e-300)
        turn = np.degrees(np.arccos(np.clip(c, -1, 1)))
        anchor = (turn > corner_deg) | protect[L] | np.isin(L, list(pinch))
        for k in np.argsort(seg):
            if seg[k] >= short_frac * h:
                continue
            a, b = k, (k + 1) % m
            if anchor[a] and anchor[b]:
                if log is not None:
                    log.append(dict(edge=(int(L[a]), int(L[b])), len=float(seg[k]), skip="2 ancrages"))
                continue
            # fenêtre [i0, i1] (indices circulaires) : s'arrête aux ancrages
            i0 = a
            for _ in range(window):
                if anchor[i0 % m]:
                    break
                i0 -= 1
            i1 = b if b > a else b + m
            for _ in range(window):
                if anchor[i1 % m]:
                    break
                i1 += 1
            idx = [j % m for j in range(i0, i1 + 1)]
            if len(idx) < 3 or len(set(idx)) != len(idx):
                if log is not None:
                    log.append(dict(edge=(int(L[a]), int(L[b])), len=float(seg[k]), skip="fenêtre vide"))
                continue
            pts = X[L[idx]].copy()
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.r_[0.0, np.cumsum(d)]
            if s[-1] <= 0:
                continue
            # pas uniforme le long de la polyligne d'origine
            t = np.linspace(0.0, s[-1], len(idx))
            new = np.empty_like(pts)
            for j, tj in enumerate(t):
                r = min(np.searchsorted(s, tj, side="right") - 1, len(d) - 1)
                w = (tj - s[r]) / max(d[r], 1e-300)
                new[j] = pts[r] + w * (pts[r + 1] - pts[r])
            new[0], new[-1] = pts[0], pts[-1]
            moved = [int(L[j]) for j in idx[1:-1]]
            # quads touchés : autour des nœuds déplacés, sur 2 anneaux
            ring = {i for n in moved for i in node_quads[n]}
            inner = set()
            for _ in range(2):
                ns = {int(n) for i in ring for n in Q[i]}
                inner |= {n for n in ns if not fixed[n]}
                ring |= {i for n in ns for i in node_quads[n]}
            ring = sorted(ring)
            # évaluation locale (pas de copie du maillage entier : pièces de 10^5-10^6 nœuds)
            loc_nodes = {int(n) for i in ring for n in Q[i]} | {o for n in inner for o in nbrs[n]}
            ids = np.array(sorted(loc_nodes), dtype=np.int64)
            pos = {int(n): j for j, n in enumerate(ids)}
            Xl = X[ids].copy()
            Ql = np.array([[pos[int(n)] for n in Q[i]] for i in ring], dtype=np.int64)

            def qual_local(XX):
                qq = quad_quality(XX, Ql)
                ar, amin, amax = _quad_shape(XX, Ql)
                return float(qq.min()), float(ar.max()), float(amin.min()), float(amax.max())

            before = qual_local(Xl)
            for n, p_new in zip(moved, new[1:-1]):
                Xl[pos[n]] = p_new
            inner_l = [(pos[n], [pos[o] for o in nbrs[n]]) for n in inner]
            for _ in range(iters):
                for j, nb in inner_l:
                    Xl[j] = Xl[nb].mean(axis=0)
            after = qual_local(Xl)
            # meilleur élancement sans dégrader le jacobien (ou jacobien meilleur à élancement tenu)
            ok = (after[0] > 0.2 or after[0] >= before[0]) and after[1] < before[1] - 1e-6 \
                and (after[1] <= max_ar or after[1] < 0.8 * before[1])
            if log is not None:
                log.append(dict(ends=[(round(float(turn[j % m]), 1), bool(protect[L[j % m]])) for j in (i0, i1)],
                                edge=(int(L[a]), int(L[b])), len=float(seg[k]), n=len(idx), before=before,
                                after=after, ok=bool(ok), pos=X[L[a]].round(1).tolist()))
            if ok:
                X[ids] = Xl
                n_done += 1
                P = X[L]
                e_out = np.roll(P, -1, 0) - P
                seg = np.linalg.norm(e_out, axis=1)
    return n_done


def structured_grid(Q: np.ndarray) -> np.ndarray | None:
    """Grille (lignes x colonnes) des nœuds d'une face maillée en transfini (quads Q, un seul
    bloc, orientation cohérente), ou None si ce n'est pas une grille. Colonnes = direction la
    plus fournie (le long de la face)."""
    Q = [list(map(int, q)) for q in Q]
    if not Q:
        return None
    edge_q = defaultdict(list)
    node_q = defaultdict(list)
    for i, q in enumerate(Q):
        for k in range(4):
            edge_q[(min(q[k], q[(k + 1) % 4]), max(q[k], q[(k + 1) % 4]))].append(i)
            node_q[q[k]].append(i)
    corner_q = [i for i, q in enumerate(Q) if sum(len(node_q[n]) == 1 for n in q) >= 1]
    if not corner_q:
        return None
    i0 = corner_q[0]
    q0 = Q[i0]
    r = next(k for k in range(4) if len(node_q[q0[k]]) == 1)
    q0 = q0[r:] + q0[:r]                       # n0 = coin

    def other(i, a, b):
        c = [j for j in edge_q[(min(a, b), max(a, b))] if j != i]
        return c[0] if c else None

    def oriented(j, first, last):
        """Quad j tourné pour que q[0] = first et q[3] = last."""
        q = Q[j]
        for k in range(4):
            qq = q[k:] + q[:k]
            if qq[0] == first and qq[3] == last:
                return qq
        return None

    pos = {i0: (0, 0)}
    quad_at = {(0, 0): q0}
    todo = [(i0, q0)]
    while todo:
        i, q = todo.pop()
        a, b = pos[i]
        for (da, db), (e0, e1), (f0, f1) in (((1, 0), (q[1], q[2]), (q[1], q[2])),
                                             ((0, 1), (q[3], q[2]), (q[3], q[2]))):
            j = other(i, e0, e1)
            if j is None or j in pos:
                continue
            qq = oriented(j, f0, f1) if (da, db) == (1, 0) else None
            if (da, db) == (0, 1):
                # voisin du dessus : son n0 = notre n3, son n1 = notre n2
                qj = Q[j]
                qq = next((qj[k:] + qj[:k] for k in range(4) if (qj[k:] + qj[:k])[0] == q[3]
                           and (qj[k:] + qj[:k])[1] == q[2]), None)
            if qq is None:
                return None
            pos[j] = (a + da, b + db)
            quad_at[pos[j]] = qq
            todo.append((j, qq))
    if len(pos) != len(Q):
        return None
    na = max(p[0] for p in pos.values()) + 1
    nb = max(p[1] for p in pos.values()) + 1
    if na * nb != len(Q):
        return None
    G = -np.ones((nb + 1, na + 1), np.int64)   # G[ligne, colonne]
    for (a, b), q in quad_at.items():
        G[b, a], G[b, a + 1], G[b + 1, a + 1], G[b + 1, a] = q[0], q[1], q[2], q[3]
    if (G < 0).any():
        return None
    return G if G.shape[1] >= G.shape[0] else G.T.copy()


def _poly_param(P: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _poly_at(P: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points de la polyligne P (abscisses s) aux abscisses t."""
    t = np.clip(t, 0.0, s[-1])
    j = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(s) - 2)
    w = (t - s[j]) / np.maximum(s[j + 1] - s[j], 1e-300)
    return P[j] + w[:, None] * (P[j + 1] - P[j])


def _poly_project(P: np.ndarray, s: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Abscisse du point de la polyligne P le plus proche de chaque point."""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.empty(len(pts))
    for k, p in enumerate(pts):
        t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        j = int(np.argmin(d))
        out[k] = s[j] + t[j] * (s[j + 1] - s[j])
    return out


def align_columns(qm: QuadMesh, faces, anchors: set, min_gap: float = 0.3,
                  min_quality: float = 0.2) -> dict:
    """Colonnes des faces transfinies redressées : les nœuds d'un grand côté glissent le long
    de ce côté (polyligne des nœuds) pour faire face à ceux de l'autre côté ; propagation de
    bande en bande (patte -> pli -> âme -> ...) ; nœuds sur un sommet CAD (anchors) fixes ;
    transition près des bouts ; nœuds intérieurs de chaque colonne déplacés par interpolation
    des déplacements des deux bouts de colonne. Le transfini relie le k-ième nœud d'un côté au
    k-ième de l'autre : deux côtés de longueurs différentes (bouts en biais d'une patte de
    upper part_001 : 7 405 et 7 330 mm) donnent des colonnes penchées sur toute la longueur.
    Tout est annulé si la qualité des quads touchés baisse. Renvoie des statistiques."""
    X0 = qm.X.copy()
    Q, F = qm.quads, qm.quad_face
    grids = {}
    for f in faces:
        g = structured_grid(Q[F == f])
        if g is not None and g.shape[0] >= 2 and g.shape[1] >= 3:
            grids[int(f)] = g
    if not grids:
        return dict(faces=0)
    # côtés = premières / dernières lignes ; côté partagé entre deux faces = même ensemble de nœuds
    side_of = defaultdict(list)                 # clé -> [(face, 0 | -1)]
    for f, g in grids.items():
        for w in (0, -1):
            side_of[frozenset(g[w].tolist())].append((f, w))
    placed: dict = {}                           # clé de côté -> tableau ordonné de nœuds (positionné)
    moved_faces = 0

    def n_anchor(line):
        return sum(1 for n in line[1:-1] if n in anchors)

    # ordre : faces dont un côté porte des sommets CAD intérieurs (contraints) d'abord
    order = sorted(grids, key=lambda f: -max(n_anchor(grids[f][0]), n_anchor(grids[f][-1])))
    queue = list(order)
    seen = set()
    while queue:
        f = queue.pop(0)
        if f in seen:
            continue
        g = grids[f]
        k0, k1 = frozenset(g[0].tolist()), frozenset(g[-1].tolist())
        if k0 in placed and k1 in placed:
            ref_w, mov_w = 0, -1
            mov_ok = False
        elif k1 in placed:
            ref_w, mov_w, mov_ok = -1, 0, True
        elif k0 in placed:
            ref_w, mov_w, mov_ok = 0, -1, True
        else:
            # aucun côté placé : le plus contraint sert de référence
            ref_w, mov_w = (0, -1) if n_anchor(g[0]) >= n_anchor(g[-1]) else (-1, 0)
            mov_ok = True
        seen.add(f)
        ref, mov = g[ref_w], g[mov_w]
        placed[frozenset(ref.tolist())] = ref
        if mov_ok:
            P = qm.X[mov].copy()
            s = _poly_param(P)
            L = s[-1]
            n = len(mov) - 1
            if L > 0 and n >= 2:
                s_star = _poly_project(P, s, qm.X[ref])
                width = float(np.mean(np.linalg.norm(qm.X[g[0]] - qm.X[g[-1]], axis=1)))
                e0, e1 = abs(s_star[0] - s[0]), abs(s_star[-1] - s[-1])
                Lt = 2.0 * width + 3.0 * max(e0, e1)
                sr = _poly_param(qm.X[ref])
                d = np.minimum(sr, sr[-1] - sr)
                wgt = np.clip(d / max(Lt, 1e-9), 0.0, 1.0)
                t = wgt * s_star + (1.0 - wgt) * s
                # nœuds sur un sommet CAD et extrémités : fixes ; ordre strict, écart minimal
                fixed_k = [0, n] + [k for k in range(1, n) if mov[k] in anchors]
                t[fixed_k] = s[fixed_k]
                gap = min_gap * L / n
                ok = True
                for a_, b_ in zip(sorted(fixed_k)[:-1], sorted(fixed_k)[1:]):
                    seg = t[a_:b_ + 1].copy()
                    for k in range(1, len(seg)):
                        seg[k] = max(seg[k], seg[k - 1] + gap)
                    for k in range(len(seg) - 2, -1, -1):
                        seg[k] = min(seg[k], seg[k + 1] - gap)
                    if seg[0] < t[a_] - 1e-9 or seg[-1] > t[b_] + 1e-9 or np.any(np.diff(seg) <= 0):
                        ok = False
                        break
                    t[a_ + 1:b_] = seg[1:-1]
                if ok:
                    newP = _poly_at(P, s, t)
                    qm.X[mov[1:-1]] = newP[1:-1]
                    moved_faces += 1
            placed[frozenset(mov.tolist())] = mov
        # voisines par le côté déplacé / de référence
        for key in (frozenset(g[0].tolist()), frozenset(g[-1].tolist())):
            for (f2, _) in side_of[key]:
                if f2 not in seen:
                    queue.insert(0, f2)
    # nœuds intérieurs : déplacement interpolé le long de chaque colonne
    for f, g in grids.items():
        D0, D1 = qm.X[g[0]] - X0[g[0]], qm.X[g[-1]] - X0[g[-1]]
        C = X0[g]
        seg = np.linalg.norm(np.diff(C, axis=0), axis=2)
        frac = np.vstack([np.zeros(g.shape[1]), np.cumsum(seg, axis=0)]) / np.maximum(seg.sum(0), 1e-300)
        inner = g[1:-1]
        if len(inner):
            fr = frac[1:-1][:, :, None]
            qm.X[inner] = X0[inner] + (1.0 - fr) * D0[None] + fr * D1[None]
    # contrôle : qualité des quads des faces touchées
    sel = np.isin(F, list(grids))
    q_before = quad_quality(X0, Q[sel])
    q_after = quad_quality(qm.X, Q[sel])
    worse = (q_after.min() < min(q_before.min(), min_quality)) or \
        ((q_after < min_quality).sum() > (q_before < min_quality).sum())
    if worse:
        qm.X[:] = X0
        return dict(faces=len(grids), moved=0, reverted=True,
                    q_min_before=float(q_before.min()), q_min_after=float(q_after.min()))
    return dict(faces=len(grids), moved=moved_faces, reverted=False,
                q_min_before=float(q_before.min()), q_min_after=float(q_after.min()))
