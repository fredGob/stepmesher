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
                gain: float = 0.05, shape: dict | None = None) -> int:
    """Bascule + lissage local des quads médiocres.

    `threshold` : jacobien 2D en dessous duquel un quad est visé. `shape` (optionnel,
    seuils `max_angle_deg` / `min_angle_deg` / `max_aspect_ratio`) vise en plus les
    quads qui violent les critères de forme cibles même si leur jacobien reste au-dessus
    du seuil : un quad à 167° a un jacobien ~0.22 > 0.2 et échappait sinon à la réparation.
    """
    X = qm.X.copy()
    Q = [list(map(int, q)) for q in qm.quads]
    F = qm.quad_face.tolist()
    n_flips = 0
    for _ in range(passes):
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
            q1 = Q[i]
            cur_q = float(qual[i])
            best = None
            # option 0 : simple lissage local, sans changement de topologie
            qs, Xs, _ = loc.evaluate({i: q1})
            if qs > cur_q + gain:
                best = (qs, {i: q1}, Xs)
            for k in range(4):
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
