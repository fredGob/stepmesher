"""Chaîne complète : maillage SC8R, qualité, export et relecture .inp."""
import numpy as np
import pytest

from stepmesher.io.inp_validator import read_mesh, validate_inp
from stepmesher.mesh.quality import HEX_CORNERS, hex_internal_jacobian, scaled_jacobian
from stepmesher.testdata.synth import EXPECTED

CONSTANT = [n for n, e in EXPECTED.items() if e["kind"] == "constant"]


@pytest.mark.parametrize("name", CONSTANT)
def test_constant_parts_mesh_ok(meshed, name):
    _, reps = meshed
    assert reps[name]["status"] == "OK", reps[name].get("reasons")


def test_variable_part_is_flagged_approximate(meshed):
    _, reps = meshed
    r = reps["plaque_poches"]
    assert r["status"] == "OK_APPROX"
    assert r["approximate"] is True
    assert len(r["sections"]) == 2                      # une section par palier (2 et 6 mm)


def test_massive_part_falls_back_to_c3d10(meshed):
    out_dir, reps = meshed
    r = reps["bloc_massif"]
    assert r["status"] == "OK_TET"
    assert r["metrics"]["element_type"] == "C3D10"
    assert r["metrics"]["n_negative_volume"] == 0
    assert r["metrics"]["volume_rel_error"] < 0.01          # bloc 50x40x30
    v = validate_inp(out_dir / "bloc_massif.inp")
    assert v["ok"], v["errors"]
    mesh = read_mesh(out_dir / "bloc_massif.inp")
    et, conn = next(iter(mesh["elems"].values()))
    assert et == "C3D10" and len(conn) == 10


def test_c3d10_midnodes_abaqus_order(meshed):
    """Nœuds 5..10 au milieu des arêtes (1,2) (2,3) (3,1) (1,4) (2,4) (3,4) : convention Abaqus."""
    out_dir, _ = meshed
    mesh = read_mesh(out_dir / "bloc_massif.inp")
    X = {i: np.array(p) for i, p in mesh["nodes"].items()}
    for _, (et, c) in list(mesh["elems"].items())[:200]:
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
            mid = 0.5 * (X[c[a]] + X[c[b]])
            assert np.linalg.norm(X[c[4 + k]] - mid) < 1e-6 + 0.05 * np.linalg.norm(X[c[a]] - X[c[b]])


def test_internal_jacobian_detects_hidden_inversion():
    X = np.array([
        [1.1832, 2.0311, -0.0427], [1.0382, -0.5758, -0.0572],
        [2.8826, -0.1047, -0.1554], [1.7720, 1.6870, 0.3499],
        [-0.4955, 0.1389, 0.2733], [1.1576, -1.9515, 1.7535],
        [1.3558, 0.9367, 0.5706], [-0.9842, 1.8005, -0.2882],
    ], float)
    conn = np.arange(8, dtype=np.int64).reshape(1, 8)
    assert scaled_jacobian(X, conn, HEX_CORNERS)[0] > 0
    assert hex_internal_jacobian(X, conn)[0] < 0


@pytest.mark.parametrize("name", CONSTANT)
def test_quality_criteria(meshed, name):
    _, reps = meshed
    m = reps[name]["metrics"]
    assert m["triangle_pct"] == 0.0                     # 100 % quads
    assert m["n_sc6r"] == 0
    assert m["n_negative_jacobian"] == 0
    assert m["scaled_jacobian"]["min"] > 0.2
    assert m["reprojection_error"]["max"] < 0.01        # < 1 % de l'épaisseur
    assert m["thickness_deviation"]["max"] < 0.05


@pytest.mark.parametrize("name", CONSTANT)
def test_one_element_through_thickness(meshed, name):
    out_dir, reps = meshed
    mesh = read_mesh(out_dir / f"{name}.inp")
    n = len(mesh["nodes"]) // 2
    for et, conn in list(mesh["elems"].values())[:2000]:
        assert et == "SC8R"
        bot, top = conn[:4], conn[4:]
        assert all(b <= n for b in bot)                 # nœuds 1-4 : peau de référence
        assert [t - n for t in top] == bot              # 5-8 : mêmes nœuds décalés


@pytest.mark.parametrize("name", CONSTANT)
def test_element_thickness_matches_part(meshed, name):
    out_dir, _ = meshed
    mesh = read_mesh(out_dir / f"{name}.inp")
    X = np.array([mesh["nodes"][i] for i in sorted(mesh["nodes"])])
    n = len(X) // 2
    L = np.linalg.norm(X[n:] - X[:n], axis=1)
    t = EXPECTED[name]["t"]
    # onglet aux angles vifs : jusqu'à t*sqrt(3) aux coins trièdres
    assert np.median(L) == pytest.approx(t, rel=0.01)
    assert L.min() > 0.95 * t and L.max() < 1.8 * t


@pytest.mark.parametrize("name", CONSTANT + ["plaque_poches"])
def test_inp_validator_passes(meshed, name):
    out_dir, _ = meshed
    v = validate_inp(out_dir / f"{name}.inp")
    assert v["ok"], v["errors"]
    assert any("MATERIAL=TBD" in w for w in v["warnings"])


@pytest.mark.parametrize("name", CONSTANT)
def test_inp_roundtrip(meshed, name):
    out_dir, reps = meshed
    mesh = read_mesh(out_dir / f"{name}.inp")
    a = reps[name]["attempts"][-1]
    wd = out_dir / ".work" / name / ("microfix" if a.get("geometry") == "B" else "")
    n = sum(1 for x in reps[name]["attempts"][:a["index"]] if x.get("geometry") == a.get("geometry"))
    src = np.load(wd / f"attempt_{a['geometry']}{n:02d}.npz")
    X = np.array([mesh["nodes"][i] for i in range(1, len(mesh["nodes"]) + 1)])
    assert np.allclose(X, src["nodes"], rtol=0, atol=1e-6)
    conn = np.array([mesh["elems"][i][1] for i in range(1, len(mesh["elems"]) + 1)]) - 1
    assert np.array_equal(conn, src["hexa"])
    assert scaled_jacobian(X, conn, HEX_CORNERS).min() > 0


def test_tray_corner_miter(meshed):
    """Coin trièdre du bac : onglet -> jacobien normalisé minimal ~ 1/sqrt(3), jamais retourné."""
    _, reps = meshed
    sj = reps["bac_angles_vifs"]["metrics"]["scaled_jacobian"]["min"]
    assert sj == pytest.approx(1 / np.sqrt(3), rel=0.02)


def test_outputs_present(meshed):
    out_dir, reps = meshed
    for name in CONSTANT:
        for ext in (".inp", ".vtu", "_orientation.csv", ".json", ".log"):
            assert (out_dir / f"{name}{ext}").exists(), f"{name}{ext}"


def test_orientation_axes_orthonormal(meshed):
    out_dir, _ = meshed
    import csv
    with open(out_dir / "corniere_pliee_orientation.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows[::50]:
        s = np.array([float(r[f"stack_{c}"]) for c in "xyz"])
        a = np.array([float(r[f"axis1_{c}"]) for c in "xyz"])
        assert abs(np.linalg.norm(s) - 1) < 1e-6 and abs(np.linalg.norm(a) - 1) < 1e-6
        assert abs(s @ a) < 1e-6


@pytest.mark.parametrize("name", CONSTANT + ["plaque_poches", "bloc_massif"])
def test_inp_is_pure_ascii(meshed, name):
    """Abaqus peut refuser les caractères non ASCII, même en commentaire."""
    out_dir, _ = meshed
    data = (out_dir / f"{name}.inp").read_bytes()
    assert all(b < 128 for b in data)


def _surface_faces(inp, name):
    """Faces (liste de connectivités de face) d'une *SURFACE TYPE=ELEMENT relue dans le .inp."""
    from stepmesher.io.inp_validator import read_inp
    blocks = read_inp(inp)["blocks"]
    mesh = read_mesh(inp)
    elsets = {b["prm"]["ELSET"].upper(): None for b in blocks if b["kw"] == "ELEMENT" and "ELSET" in b["prm"]}
    surf = next(b for b in blocks if b["kw"] == "SURFACE" and b["prm"].get("NAME", "").upper() == name)
    out = []
    for ln in surf["data"]:
        es, face = (x.strip().upper() for x in ln.split(","))
        assert es in elsets
        et = "SC8R" if es == "ES_SC8R" else "SC6R"
        n = 4 if et == "SC8R" else 3
        for i, (t, conn) in mesh["elems"].items():
            if t == et:
                out.append((i, conn[:n] if face == "S1" else conn[n:]))
    return mesh, out


def test_inner_outer_skin_surfaces_on_bracket(meshed):
    """Cornière pliée (rayon intérieur 4, épaisseur 2, axe z) : SURF_INNER sur le rayon 4,
    SURF_OUTER sur le rayon 6 dans le pli, et chaque surface couvre tous les éléments."""
    out_dir, reps = meshed
    inp = out_dir / "corniere_pliee.inp"
    for name, r_expected in (("SURF_INNER", 4.0), ("SURF_OUTER", 6.0)):
        mesh, faces = _surface_faces(inp, name)
        assert len(faces) == len(mesh["elems"])
        X = mesh["nodes"]
        radii = []
        for _, fn in faces:
            P = np.array([X[k] for k in fn])
            if (P[:, 0] > 0.1).all() and (P[:, 1] > 0.1).all():     # dans le quart d'anneau du pli
                radii.append(np.hypot(P[:, 0], P[:, 1]))
        radii = np.concatenate(radii)
        assert len(radii) > 0
        assert np.allclose(radii, r_expected, atol=0.02), (name, radii.min(), radii.max())
    assert reps["corniere_pliee"]["sets"]["surfaces"]["SURF_INNER"] in ("S1", "S2")
