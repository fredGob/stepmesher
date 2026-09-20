"""Configuration, empreinte, isolation des essais, performance."""
import math
import time

import pytest

from stepmesher.config import load_config
from stepmesher.occ.loader import gmsh_session
from stepmesher.strategy.recipe import Recipe
from stepmesher.strategy.runner import run_isolated


# ------------------------------------------------------------------ config
def test_unknown_config_key_rejected():
    with pytest.raises(KeyError):
        load_config(overrides={"mesh": {"taille": 3}})


def test_hole_threshold_is_min_of_nonzero():
    c = load_config(overrides={"holes": {"max_diameter_mm": 12.0, "max_diameter_frac": 0.02}})
    assert c.hole_threshold(300.0) == pytest.approx(6.0)
    assert c.hole_threshold(3000.0) == pytest.approx(12.0)
    c = load_config(overrides={"holes": {"fill": False}})
    assert c.hole_threshold(300.0) == 0.0


def test_target_size_bounded_by_thickness_and_limits():
    c = load_config()
    # 12 m de diagonale, 0,4 mm : plafonné par k * t
    assert c.target_size(12000.0, 0.4) == pytest.approx(4.0)
    # borne max
    assert c.target_size(12000.0, 10.0) == pytest.approx(c["mesh"]["max_size_mm"])
    # borne min
    assert c.target_size(10.0, 0.01) == pytest.approx(c["mesh"]["min_size_mm"])


def test_recipe_bounds():
    r = Recipe(strategy="conform", size_mult=50, n_per_bend=1).clamp()
    assert r.size_mult == 3.0 and r.n_per_bend == 2
    with pytest.raises(ValueError):
        Recipe(strategy="inconnue").clamp()


# ------------------------------------------------------------------ empreinte
def _hash_of(path, cfg):
    from stepmesher.analyze.fingerprint import exact_hash, exact_invariants
    from stepmesher.occ.loader import import_step
    with gmsh_session(cfg):
        import_step(str(path), cfg)
        return exact_hash(exact_invariants())


def test_exact_hash_invariant_under_rigid_motion(tmp_path, cfg):
    from stepmesher.testdata.synth import write_part
    a = write_part("corniere_pliee", tmp_path / "a.step")
    b = write_part("corniere_pliee", tmp_path / "b.step",
                   transform=(0.7, (0.3, -0.5, 0.8), (1234.5, -98.7, 456.0)))
    c = write_part("profile_u", tmp_path / "c.step")
    ha, hb, hc = _hash_of(a, cfg), _hash_of(b, cfg), _hash_of(c, cfg)
    assert ha == hb
    assert ha != hc


# ------------------------------------------------------------------ isolation
def _crash(out):
    import os
    os._exit(11)


def _sleep(out):
    time.sleep(60)


def _ok(out):
    import json
    from pathlib import Path
    Path(out).write_text(json.dumps(dict(status="passed", passed=True)))


def test_runner_crash_is_contained(tmp_path):
    r = run_isolated("tests.test_infra:_crash", dict(out=str(tmp_path / "r.json")), tmp_path / "r.json", 30)
    assert r["exec_status"] == "crash" and not r["passed"]


def test_runner_timeout(tmp_path):
    t0 = time.time()
    r = run_isolated("tests.test_infra:_sleep", dict(out=str(tmp_path / "r.json")), tmp_path / "r.json", 2)
    assert r["exec_status"] == "timeout" and time.time() - t0 < 20


def test_runner_ok(tmp_path):
    r = run_isolated("tests.test_infra:_ok", dict(out=str(tmp_path / "r.json")), tmp_path / "r.json", 30)
    assert r["exec_status"] == "ok" and r["passed"]


# ------------------------------------------------------------------ performance
def test_analysis_scales_on_perforated_plate(tmp_path, cfg):
    """Plaque percée de 300 trous : l'analyse doit rester rapide (garde-fou O(n²))."""
    import gmsh
    from stepmesher.analyze.pipeline import prepare_part
    step = tmp_path / "perf.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        b = occ.addBox(0, 0, 0, 600, 400, 2.0)
        holes = [occ.addCylinder(20 + 30 * i, 20 + 30 * j, -1, 0, 0, 4, 2.5 if (i + j) % 2 else 6.0)
                 for i in range(19) for j in range(13)]
        occ.cut([(3, b)], [(3, h) for h in holes])
        occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()
    t0 = time.time()
    with gmsh_session(cfg):
        pa = prepare_part(step, tmp_path / "w", cfg)
    dt = time.time() - t0
    assert pa.kind == "constant"
    assert pa.classification["t_median"] == pytest.approx(2.0, rel=0.01)
    n_holes = len(pa.holes_filled) + len(pa.holes_kept)
    assert n_holes == 19 * 13
    assert dt < 180, f"analyse trop lente : {dt:.0f} s"
    assert math.isfinite(pa.diag)
