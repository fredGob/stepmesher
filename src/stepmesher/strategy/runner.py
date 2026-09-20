"""Exécution isolée : chaque tâche gmsh tourne dans un processus enfant avec délai.

Un segfault ou un blocage de gmsh devient un simple essai échoué, avec sa raison.
Le résultat transite par un fichier JSON écrit par l'enfant.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path


def _child(target: str, kwargs: dict):
    import importlib
    import logging
    logging.basicConfig(level=logging.WARNING)
    mod, fn = target.rsplit(":", 1)
    getattr(importlib.import_module(mod), fn)(**kwargs)


def run_isolated(target: str, kwargs: dict, result_json: Path, timeout: float) -> dict:
    """Lance `module:fonction(**kwargs)` dans un processus « spawn ».

    La fonction doit écrire `result_json`. Retourne ce JSON, enrichi du statut
    d'exécution ('ok' | 'timeout' | 'crash').
    """
    result_json = Path(result_json)
    if result_json.exists():
        result_json.unlink()
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child, args=(target, kwargs), daemon=False)
    t0 = time.time()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join(10)
        return dict(exec_status="timeout", passed=False, status="timeout",
                    reasons=[f"délai dépassé ({timeout:.0f} s)"], timings=dict(total=time.time() - t0))
    if result_json.exists():
        try:
            d = json.loads(result_json.read_text())
            d["exec_status"] = "ok" if p.exitcode == 0 else "crash"
            return d
        except json.JSONDecodeError:
            pass
    return dict(exec_status="crash", passed=False, status="crash",
                reasons=[f"processus terminé anormalement (code {p.exitcode})"], timings=dict(total=time.time() - t0))
