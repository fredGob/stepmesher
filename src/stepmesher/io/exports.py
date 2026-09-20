"""Exports complémentaires : .vtu (ParaView) et CSV d'orientation par élément."""
from __future__ import annotations

from pathlib import Path

import numpy as np

VTK_HEXA, VTK_WEDGE = 12, 13


def write_vtu(path: Path, mesh: dict, zone_of: np.ndarray | None = None):
    X = mesh["nodes"]
    H, W = mesh["hexa"], mesh["wedge"]
    conn = np.r_[H.ravel(), W.ravel()]
    offsets = np.r_[np.arange(1, len(H) + 1) * 8, len(H) * 8 + np.arange(1, len(W) + 1) * 6]
    types = np.r_[np.full(len(H), VTK_HEXA), np.full(len(W), VTK_WEDGE)]
    cells = dict(
        thickness=mesh["thickness"], scaled_jacobian=mesh["sj"],
        cad_face=np.r_[mesh["hexa_face"], mesh["wedge_face"]].astype(float),
        quality_warn=(np.asarray(mesh["soft"], float) if "soft" in mesh else np.zeros(len(types))),
    )
    if zone_of is not None:
        cells["zone"] = zone_of.astype(float) + 1
    vecs = dict(normal=mesh["normal"], stack_direction=mesh["stack"], axis1=mesh["axis1"])

    def arr(a):
        return " ".join(f"{x:.7g}" for x in np.asarray(a).ravel())

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<UnstructuredGrid><Piece NumberOfPoints="{len(X)}" NumberOfCells="{len(types)}">\n')
        f.write('<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(arr(X) + "\n</DataArray></Points>\n<Cells>\n")
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n' + " ".join(map(str, conn)) + "\n</DataArray>\n")
        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n' + " ".join(map(str, offsets)) + "\n</DataArray>\n")
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n' + " ".join(map(str, types)) + "\n</DataArray>\n")
        f.write("</Cells>\n<CellData>\n")
        for k, v in cells.items():
            f.write(f'<DataArray type="Float64" Name="{k}" format="ascii">\n{arr(v)}\n</DataArray>\n')
        for k, v in vecs.items():
            f.write(f'<DataArray type="Float64" Name="{k}" NumberOfComponents="3" format="ascii">\n{arr(v)}\n</DataArray>\n')
        f.write("</CellData>\n</Piece></UnstructuredGrid>\n</VTKFile>\n")


def write_orientation_csv(path: Path, mesh: dict, zone_names: list[str] | None = None, zone_of=None):
    n = len(mesh["thickness"])
    with open(path, "w") as f:
        f.write("element,normal_x,normal_y,normal_z,stack_x,stack_y,stack_z,axis1_x,axis1_y,axis1_z,thickness,zone\n")
        for k in range(n):
            nv, s, a = mesh["normal"][k], mesh["stack"][k], mesh["axis1"][k]
            z = zone_names[zone_of[k]] if zone_names is not None and zone_of is not None else ""
            f.write(f"{k + 1},{nv[0]:.6g},{nv[1]:.6g},{nv[2]:.6g},{s[0]:.6g},{s[1]:.6g},{s[2]:.6g},"
                    f"{a[0]:.6g},{a[1]:.6g},{a[2]:.6g},{mesh['thickness'][k]:.6g},{z}\n")


def write_vtu_tet(path: Path, mesh: dict):
    X, C = mesh["nodes"], mesh["conn"]
    npe = C.shape[1]
    vtype = 24 if npe == 10 else 10         # quadratic tetra / tetra (même ordre qu'Abaqus)

    def arr(a):
        return " ".join(f"{x:.7g}" for x in np.asarray(a).ravel())

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<UnstructuredGrid><Piece NumberOfPoints="{len(X)}" NumberOfCells="{len(C)}">\n')
        f.write('<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n' + arr(X) + "\n</DataArray></Points>\n<Cells>\n")
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n' + " ".join(map(str, C.ravel())) + "\n</DataArray>\n")
        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n' + " ".join(map(str, np.arange(1, len(C) + 1) * npe)) + "\n</DataArray>\n")
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n' + " ".join([str(vtype)] * len(C)) + "\n</DataArray>\n")
        f.write("</Cells>\n<CellData>\n")
        f.write('<DataArray type="Float64" Name="shape_quality" format="ascii">\n' + arr(mesh["quality"]) + "\n</DataArray>\n")
        f.write("</CellData>\n</Piece></UnstructuredGrid>\n</VTKFile>\n")
