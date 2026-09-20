from __future__ import annotations

from pathlib import Path

import pytest

from stepmesher.config import load_config
from stepmesher.occ.loader import gmsh_session

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "stps_test"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    from stepmesher.testdata.synth import generate_all
    d = tmp_path_factory.mktemp("synth")
    generate_all(d)
    return d


@pytest.fixture(scope="session")
def analyses(synth_dir, cfg, tmp_path_factory):
    """Analyse de chaque pièce synthétique (une fois par session)."""
    from stepmesher.analyze.pipeline import prepare_part
    out = {}
    work = tmp_path_factory.mktemp("work")
    for step in sorted(synth_dir.glob("*.step")):
        with gmsh_session(cfg):
            out[step.stem] = prepare_part(step, work / step.stem, cfg)
    return out


@pytest.fixture(scope="session")
def meshed(synth_dir, cfg, tmp_path_factory):
    """Maillage complet (CLI interne) de chaque pièce synthétique."""
    from stepmesher.process import process_part
    out_dir = tmp_path_factory.mktemp("out")
    reps = {}
    for step in sorted(synth_dir.glob("*.step")):
        reps[step.stem] = process_part(step, out_dir, cfg, keep_work=True)
    return out_dir, reps
