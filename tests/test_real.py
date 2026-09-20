"""Pièces réelles (stps_test/) : lent, lancé avec `pytest -m real`."""
import pytest

from stepmesher.occ.loader import gmsh_session

from .conftest import REAL_DIR

pytestmark = [pytest.mark.real,
              pytest.mark.skipif(not REAL_DIR.exists(), reason="stps_test/ absent")]


def _step(prefix):
    return next(REAL_DIR.glob(f"{prefix}*.stp"))


def test_part_000_is_constant_sheet(cfg, tmp_path):
    from stepmesher.analyze.pipeline import prepare_part
    with gmsh_session(cfg):
        pa = prepare_part(_step("part_000"), tmp_path, cfg)
    assert pa.kind == "constant"
    assert pa.classification["t_median"] == pytest.approx(4.34, rel=0.02)
    assert len(pa.bends) == 3


@pytest.mark.parametrize("prefix,zones", [("part_349", 3)])
def test_chem_milled_steps_detected(cfg, tmp_path, prefix, zones):
    from stepmesher.analyze.pipeline import prepare_part
    with gmsh_session(cfg):
        pa = prepare_part(_step(prefix), tmp_path, cfg)
    assert pa.kind == "variable"
    assert len(pa.thickness_zones) >= zones


def test_part_000_meshes_ok(cfg, tmp_path):
    from stepmesher.io.inp_validator import validate_inp
    from stepmesher.process import process_part
    rep = process_part(_step("part_000"), tmp_path, cfg)
    assert rep["status"] == "OK", rep.get("reasons")
    assert validate_inp(rep["output"])["ok"]


@pytest.mark.parametrize("prefix,expected", [("part_093", "OK_TET"), ("part_145", "OK_APPROX"),
                                             ("part_201", "OK_APPROX"), ("part_349", "OK_APPROX")])
def test_other_parts_mesh(cfg, tmp_path, prefix, expected):
    """Les 4 autres pièces : épaisseur variable en SC8R approché, part_093 en C3D10 (plusieurs minutes chacune)."""
    from stepmesher.io.inp_validator import validate_inp
    from stepmesher.process import process_part
    rep = process_part(_step(prefix), tmp_path, cfg)
    assert rep["status"] == expected, rep.get("reasons")
    assert validate_inp(rep["output"])["ok"]
