"""Outil de non-régression entre campagnes (tools/compare_runs.py)."""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("compare_runs", Path(__file__).parents[1] / "tools" / "compare_runs.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def _rep(d: Path, part, status, n_el, sj=0.5, enc="utf-8"):
    r = dict(part=part, status=status, metrics=dict(n_elements=n_el, scaled_jacobian=dict(min=sj)),
             analysis=dict(kind="constant"), timings=dict(total=1.0), note="é")
    d.mkdir(exist_ok=True)
    (d / f"{part}.json").write_bytes(json.dumps(r, ensure_ascii=False).encode(enc))


def test_compare_flags(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _rep(a, "p1", "OK", 1000, enc="cp1252")
    _rep(b, "p1", "OK", 1050)
    _rep(a, "p2", "OK", 1000)
    _rep(b, "p2", "FAILED_QUALITY", 1000)
    _rep(a, "p3", "OK_TET", 1000)
    _rep(b, "p3", "OK", 500, sj=0.1)
    rows = {r["part"]: r["flags"] for r in cr.compare(cr.load_run(a), cr.load_run(b))}
    assert rows["p1"] == []
    assert "REGRESSION statut" in rows["p2"]
    assert "gain statut" in rows["p3"] and not any(f.startswith("REGRESSION") for f in rows["p3"])
    assert cr.main([str(a), str(b)]) == 1


def test_compare_topologie(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, n_irr in ((a, 40), (b, 150)):
        d.mkdir()
        r = dict(part="p1", status="OK", analysis=dict(kind="constant"), timings=dict(total=1.0),
                 metrics=dict(n_elements=1000, scaled_jacobian=dict(min=0.5), topology=dict(n_irregular=n_irr, n_faces=2)))
        (d / "p1.json").write_text(json.dumps(r), encoding="utf-8")
    rows = {r["part"]: r["flags"] for r in cr.compare(cr.load_run(a), cr.load_run(b))}
    assert any(f.startswith("REGRESSION topologie") for f in rows["p1"])
    rows = {r["part"]: r["flags"] for r in cr.compare(cr.load_run(b), cr.load_run(a))}
    assert any(f.startswith("gain topologie") for f in rows["p1"])
