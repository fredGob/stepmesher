"""Assemblage des éléments SC8R (et SC6R optionnels), orientation par élément."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .offset import OffsetResult
from .quad import QuadMesh


@dataclass
class SolidShellMesh:
    nodes: np.ndarray        # (2N,3) : [peau de référence ; peau opposée]
    n_ref: int               # N
    hexa: np.ndarray         # (H,8) indices 0-based
    wedge: np.ndarray        # (W,6)
    hexa_face: np.ndarray    # (H,) face CAD de référence
    wedge_face: np.ndarray   # (W,)
    normal: np.ndarray       # (H+W,3) normale de la face de référence (vers la peau opposée)
    stack: np.ndarray        # (H+W,3) direction d'empilement (référence -> opposée)
    axis1: np.ndarray        # (H+W,3) axe local 1
    axis2: np.ndarray        # (H+W,3) axe local 2
    thickness: np.ndarray    # (H+W,) épaisseur moyenne de l'élément


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def build_solid_shell(qm: QuadMesh, off: OffsetResult, ref_axis) -> SolidShellMesh:
    N = len(qm.X)
    nodes = np.vstack([qm.X, off.X_top])
    X = qm.X
    Q = qm.quads.copy()
    R = qm.tris.copy()
    # orientation : la normale de 1-2-3-4 doit pointer vers la peau opposée
    if len(Q):
        nq = np.cross(X[Q[:, 2]] - X[Q[:, 0]], X[Q[:, 3]] - X[Q[:, 1]])
        dq = off.direction[Q].mean(axis=1)
        flip = np.einsum("ij,ij->i", nq, dq) < 0
        Q[flip] = Q[flip][:, [0, 3, 2, 1]]
    if len(R):
        nr = np.cross(X[R[:, 1]] - X[R[:, 0]], X[R[:, 2]] - X[R[:, 0]])
        dr = off.direction[R].mean(axis=1)
        flip = np.einsum("ij,ij->i", nr, dr) < 0
        R[flip] = R[flip][:, [0, 2, 1]]
    hexa = np.hstack([Q, Q + N])
    wedge = np.hstack([R, R + N])

    def frames(bot):
        nb = bot.shape[1]
        if len(bot) == 0:
            z = np.zeros((0, 3))
            return z, z, z, z, np.zeros(0)
        Xb = nodes[bot]
        Xt = nodes[bot + N]
        if nb == 4:
            n = np.cross(Xb[:, 2] - Xb[:, 0], Xb[:, 3] - Xb[:, 1])
        else:
            n = np.cross(Xb[:, 1] - Xb[:, 0], Xb[:, 2] - Xb[:, 0])
        n = _unit(n)
        s = _unit((Xt - Xb).mean(axis=1))
        a = np.asarray(ref_axis, float)
        a1 = a[None, :] - np.einsum("ij,j->i", s, a)[:, None] * s
        bad = np.linalg.norm(a1, axis=1) < 0.3
        if bad.any():
            alt = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            a1[bad] = alt[None, :] - np.einsum("ij,j->i", s[bad], alt)[:, None] * s[bad]
        a1 = _unit(a1)
        a2 = np.cross(s, a1)
        th = np.linalg.norm(Xt - Xb, axis=2).mean(axis=1)
        return n, s, a1, a2, th

    n1, s1, a11, a21, t1 = frames(Q)
    n2, s2, a12, a22, t2 = frames(R)
    return SolidShellMesh(nodes=nodes, n_ref=N, hexa=hexa, wedge=wedge, hexa_face=qm.quad_face,
                          wedge_face=qm.tri_face, normal=np.vstack([n1, n2]), stack=np.vstack([s1, s2]),
                          axis1=np.vstack([a11, a12]), axis2=np.vstack([a21, a22]), thickness=np.r_[t1, t2])
