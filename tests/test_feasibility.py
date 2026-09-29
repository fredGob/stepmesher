"""Score de faisabilité SC8R a priori (analyze/feasibility) — déterministe, sans maillage."""
from types import SimpleNamespace

from stepmesher.analyze.feasibility import assess_feasibility


def _pa(kind, n_ref, n_opp, n_flank, saf, n_failed=0):
    return SimpleNamespace(
        kind=kind, ref_faces=list(range(n_ref)), opp_faces=list(range(1000, 1000 + n_opp)),
        flank_faces=list(range(2000, 2000 + n_flank)), failed_faces=list(range(n_failed)),
        classification={"skin_area_fraction": saf})


def test_balanced_shell_is_sc8r():
    r = assess_feasibility(_pa("constant", 7, 7, 20, 0.75))
    assert r["verdict"] == "sc8r"
    assert r["score"] >= 0.55


def test_massive_or_no_opposite_is_tet():
    assert assess_feasibility(_pa("massive", 0, 0, 208, 0.0))["verdict"] == "tet"
    assert assess_feasibility(_pa("variable", 10, 0, 50, 0.4))["verdict"] == "tet"
    assert assess_feasibility(_pa("variable", 10, 0, 50, 0.4))["score"] == 0.0


def test_low_skin_coverage_is_tet():
    r = assess_feasibility(_pa("variable", 5, 60, 300, 0.30))
    assert r["verdict"] == "tet"
    assert r["score"] < 0.40


def test_score_bounded_and_signals_present():
    r = assess_feasibility(_pa("variable", 24, 49, 354, 0.55))
    assert 0.0 <= r["score"] <= 1.0
    assert set(r["signals"]) >= {"skin_area_fraction", "side_balance", "failed_frac"}


def test_failed_faces_penalize_score():
    clean = assess_feasibility(_pa("constant", 7, 7, 20, 0.75))["score"]
    faulty = assess_feasibility(_pa("constant", 7, 7, 20, 0.75, n_failed=10))["score"]
    assert faulty < clean
