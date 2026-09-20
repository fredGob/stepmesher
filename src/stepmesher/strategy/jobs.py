"""Tâches exécutées en sous-processus (point d'entrée `module:fonction`)."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from ..analyze.pipeline import prepare_part
from ..config import Config
from ..occ.loader import gmsh_session


def prepare_job(step: str, workdir: str, cfg_data: dict, reference_skin: str | None, out_json: str):
    out = Path(out_json)
    try:
        cfg = Config(cfg_data)
        with gmsh_session(cfg):
            pa = prepare_part(step, workdir, cfg, reference_skin)
        pa.save(Path(workdir) / "analysis.json")
        out.write_text(json.dumps(dict(status="ok", analysis=str(Path(workdir) / "analysis.json"))))
    except Exception as e:  # noqa: BLE001
        out.write_text(json.dumps(dict(status="error", reasons=[f"{type(e).__name__}: {e}"],
                                       traceback=traceback.format_exc()[-3000:])))
