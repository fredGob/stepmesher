"""Triangulation d'analyse orientée et adjacence entre faces.

Toute l'analyse géométrique travaille sur une triangulation grossière et conforme
de la peau du solide, orientée vers l'extérieur par propagation topologique puis
par le signe du volume : aucune requête OCC point par point (point intérieur,
projection), ce qui garde l'analyse en O(N log N) sur des pièces de 1000+ faces.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import gmsh
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SurfTri:
    P: np.ndarray            # (N,3) sommets
    T: np.ndarray            # (M,3) triangles, orientés vers l'extérieur
    face: np.ndarray         # (M,) tag de face CAD de chaque triangle
    normal: np.ndarray       # (M,3) normale sortante unitaire
    area: np.ndarray         # (M,)
    centroid: np.ndarray     # (M,3)
    size: np.ndarray         # (M,) plus grande arête
    face_tags: list[int]
    failed_faces: list[int] = field(default_factory=list)
    free_edges: int = 0
    nonmanifold_edges: int = 0
    # adjacence : (fa, fb) fa<fb -> dict(angle_deg, length, convex)
    adjacency: dict = field(default_factory=dict)
    # arêtes de maillage entre deux triangles de faces différentes
    edge_tris: np.ndarray | None = None     # (E,2) indices des 2 triangles
    edge_len: np.ndarray | None = None      # (E,)
    seconds: float = 0.0

    def face_area(self) -> dict[int, float]:
        u, inv = np.unique(self.face, return_inverse=True)
        s = np.bincount(inv, weights=self.area, minlength=len(u))
        return {int(f): float(a) for f, a in zip(u, s)}

    def neighbors(self) -> dict[int, set[int]]:
        nb: dict[int, set[int]] = defaultdict(set)
        for (a, b) in self.adjacency:
            nb[a].add(b)
            nb[b].add(a)
        return nb


def build_analysis_mesh(size: float, curvature: int = 12, min_factor: float = 0.02) -> SurfTri:
    """Triangule toutes les faces du modèle courant (Frontal-Delaunay, grossier)."""
    t0 = time.time()
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size * min_factor)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
    try:
        gmsh.model.mesh.generate(2)
    except Exception as e:  # noqa: BLE001 - des faces peuvent échouer, on continue
        log.warning("triangulation d'analyse : %s", e)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    P = coords.reshape(-1, 3)
    idx = np.full(int(node_tags.max()) + 1 if len(node_tags) else 1, -1, dtype=np.int64)
    idx[node_tags.astype(np.int64)] = np.arange(len(node_tags))

    tris, tface, failed, ftags = [], [], [], []
    for _, ftag in gmsh.model.getEntities(2):
        ftags.append(ftag)
        etypes, _, enodes = gmsh.model.mesh.getElements(2, ftag)
        got = False
        for et, en in zip(etypes, enodes):
            if et == 2 and len(en):
                t = idx[en.astype(np.int64)].reshape(-1, 3)
                tris.append(t)
                tface.append(np.full(len(t), ftag, dtype=np.int64))
                got = True
        if not got:
            failed.append(ftag)
    if failed:
        log.warning("%d face(s) non triangulée(s), ignorée(s) : %s", len(failed), failed[:20])
    T = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    F = np.concatenate(tface) if tface else np.zeros(0, np.int64)

    st = _orient_and_finish(P, T, F)
    st.face_tags = sorted(ftags)
    st.failed_faces = failed
    st.seconds = time.time() - t0
    return st


def _orient_and_finish(P, T, F) -> SurfTri:
    M = len(T)
    # arêtes orientées
    e0 = T[:, [0, 1, 2]].reshape(-1)
    e1 = T[:, [1, 2, 0]].reshape(-1)
    tri_of = np.repeat(np.arange(M), 3)
    lo, hi = np.minimum(e0, e1), np.maximum(e0, e1)
    fwd = e0 < e1
    key = lo.astype(np.int64) * (len(P) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)
    free_edges = int(np.sum(counts == 1))
    nonman = int(np.sum(counts > 2))

    pair_mask = counts == 2
    s = start[pair_mask]
    ta = tri_of[order[s]]
    tb = tri_of[order[s + 1]]
    consistent = fwd[order[s]] != fwd[order[s + 1]]
    edge_tris = np.stack([ta, tb], 1)
    ea, eb = e0[order[s]], e1[order[s]]
    edge_len = np.linalg.norm(P[ea] - P[eb], axis=1)
    edge_vec = P[eb] - P[ea]

    # --- orientation relative entre faces (vote par arête) ---
    fa, fb = F[ta], F[tb]
    rel = defaultdict(lambda: [0, 0])  # (f1,f2) -> [votes même sens, votes inversés]
    intra_bad = 0
    for x, y, c in zip(fa.tolist(), fb.tolist(), consistent.tolist()):
        if x == y:
            if not c:
                intra_bad += 1
            continue
        k = (x, y) if x < y else (y, x)
        rel[k][0 if c else 1] += 1
    if intra_bad:
        log.debug("%d arêtes incohérentes à l'intérieur d'une face", intra_bad)

    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (x, y), (same, inv) in rel.items():
        r = 1 if same >= inv else -1
        graph[x].append((y, r))
        graph[y].append((x, r))

    faces = np.unique(F)
    sign = {}
    tri_sign = np.ones(M)
    comp_of = {}
    ncomp = 0
    for f0 in faces.tolist():
        if f0 in sign:
            continue
        sign[f0] = 1
        comp_of[f0] = ncomp
        q = deque([f0])
        while q:
            f = q.popleft()
            for g, r in graph[f]:
                if g not in sign:
                    sign[g] = sign[f] * r
                    comp_of[g] = ncomp
                    q.append(g)
        ncomp += 1

    fs = np.array([sign[int(f)] for f in F], dtype=float) if M else np.zeros(0)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    cr = np.cross(b - a, c - a)
    vol_contrib = np.einsum("ij,ij->i", a, cr) / 6.0 * fs
    comp_arr = np.array([comp_of[int(f)] for f in F]) if M else np.zeros(0, int)
    comp_vol = np.bincount(comp_arr, weights=vol_contrib, minlength=ncomp) if M else np.zeros(0)
    flip_comp = comp_vol < 0
    fs = fs * np.where(flip_comp[comp_arr], -1.0, 1.0) if M else fs
    # triangles réorientés
    T = T.copy()
    neg = fs < 0
    T[neg] = T[neg][:, [0, 2, 1]]
    cr = cr * fs[:, None]
    area = 0.5 * np.linalg.norm(cr, axis=1)
    normal = cr / np.maximum(2 * area, 1e-300)[:, None]
    centroid = (a + b + c) / 3.0
    size = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                            np.linalg.norm(a - c, axis=1)], 1), axis=1) if M else np.zeros(0)

    # --- adjacence entre faces : angle dièdre et convexité ---
    n1, n2 = normal[ta], normal[tb]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1)))
    convex = np.einsum("ij,ij->i", centroid[tb] - centroid[ta], n1) < 0
    acc = defaultdict(lambda: [[], 0.0, 0.0])
    for x, y, an, L, cv in zip(fa.tolist(), fb.tolist(), ang.tolist(), edge_len.tolist(), convex.tolist()):
        if x == y:
            continue
        k = (x, y) if x < y else (y, x)
        acc[k][0].append(an)
        acc[k][1] += L
        acc[k][2] += L if cv else 0.0
    adjacency = {k: dict(angle_deg=float(np.percentile(v[0], 90)), length=v[1],
                         convex=v[2] / max(v[1], 1e-30)) for k, v in acc.items()}

    st = SurfTri(P=P, T=T, face=F, normal=normal, area=area, centroid=centroid, size=size,
                 face_tags=[], free_edges=free_edges, nonmanifold_edges=nonman,
                 adjacency=adjacency, edge_tris=edge_tris, edge_len=edge_len)
    st._edge_vec = edge_vec  # type: ignore[attr-defined]
    return st


def obb(P: np.ndarray, w: np.ndarray | None = None):
    """Boîte englobante orientée par ACP : (centre, axes (3,3) lignes, étendues (3,))."""
    if w is None:
        w = np.ones(len(P))
    c = np.average(P, axis=0, weights=w)
    X = P - c
    C = (X * w[:, None]).T @ X / w.sum()
    ev, V = np.linalg.eigh(C)
    axes = V[:, ::-1].T  # axe principal en premier
    proj = X @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    center = c + ((lo + hi) / 2) @ axes
    return center, axes, hi - lo
