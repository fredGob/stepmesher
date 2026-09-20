"""Analyse géométrique : épaisseurs, classification, plis, trous."""
import pytest

from stepmesher.testdata.synth import EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_kind(analyses, name):
    assert analyses[name].kind == EXPECTED[name]["kind"]


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if e["t"]])
def test_thickness_within_1pct(analyses, name):
    t = analyses[name].classification["t_median"]
    assert t == pytest.approx(EXPECTED[name]["t"], rel=0.01)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if e["kind"] != "massive"])
def test_bends(analyses, name):
    assert len(analyses[name].bends) == EXPECTED[name]["bends"]


def test_bend_radius_and_angle_exact(analyses):
    b = analyses["corniere_pliee"].bends[0]
    assert b["radius"] == pytest.approx(4.0, rel=0.02)       # rayon intérieur
    assert b["angle_deg"] == pytest.approx(90.0, abs=3.0)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if e["kind"] != "massive"])
def test_holes(analyses, name):
    pa = analyses[name]
    assert len(pa.holes_filled) == EXPECTED[name]["holes_filled"]
    assert len(pa.holes_kept) == EXPECTED[name]["holes_kept"]
    assert not pa.holes_fill_failed


def test_filled_hole_diameter(analyses):
    d = sorted(h["diameter"] for h in analyses["plaque_trouee"].holes_filled)
    assert d == pytest.approx([4.0, 4.0], rel=0.1)
    assert analyses["plaque_trouee"].holes_kept[0]["diameter"] == pytest.approx(30.0, rel=0.1)


def test_tray_sharp_edges(analyses):
    # 4 arêtes de fond + 4 arêtes verticales sur la peau intérieure
    assert len(analyses["bac_angles_vifs"].sharp_edges) == 8


def test_reference_is_inner_skin(analyses):
    for name in ("corniere_pliee", "profile_u", "bac_angles_vifs"):
        pa = analyses[name]
        a = pa.classification  # noqa: F841
        # peau intérieure = plus petite aire (cf. choose_reference_side)
        assert "intérieure" in pa.reference_reason


def test_pocket_plate_reference_is_unmachined_side(analyses):
    pa = analyses["plaque_poches"]
    assert len(pa.ref_faces) == 1                      # face plane non usinée
    assert pa.thickness_zones == pytest.approx([2.0, 6.0], rel=0.02)


def test_sides_consistent(analyses):
    for name, pa in analyses.items():
        if pa.kind == "massive":
            continue
        assert set(pa.ref_faces).isdisjoint(pa.opp_faces)
        assert pa.ref_faces and pa.opp_faces
