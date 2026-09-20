"""Interface en ligne de commande de stepmesher."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_PATH, load_config

STEP_EXT = {".stp", ".step", ".STP", ".STEP"}


def _collect(inputs: list[str], recursive: bool) -> list[Path]:
    out = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out += sorted(q for q in it if q.suffix in STEP_EXT and ".work" not in q.parts)
        elif p.exists():
            out.append(p)
        else:
            print(f"introuvable : {p}", file=sys.stderr)
    return out


def _overrides(a) -> dict:
    ov: dict = {}
    if getattr(a, "allow_wedge", None) is not None:
        ov.setdefault("quality", {})["allow_wedge_pct"] = a.allow_wedge
    if getattr(a, "no_fill_holes", False):
        ov.setdefault("holes", {})["fill"] = False
    if getattr(a, "fill_holes", None) is not None:
        ov.setdefault("holes", {}).update(fill=True, max_diameter_mm=a.fill_holes, max_diameter_frac=0.0)
    if getattr(a, "strategies", None):
        ov.setdefault("mesh", {})["strategies"] = a.strategies.split(",")
    if getattr(a, "threads", None):
        ov.setdefault("general", {})["threads"] = a.threads
    if getattr(a, "tet_order", None):
        ov.setdefault("tet", {})["order"] = a.tet_order
    if getattr(a, "llm", False):
        ov.setdefault("llm", {})["enabled"] = True
    if getattr(a, "no_llm", False):
        ov.setdefault("llm", {})["enabled"] = False
    if getattr(a, "llm_url", None):
        ov.setdefault("llm", {}).update(enabled=True, base_url=a.llm_url)
    if getattr(a, "no_tet", False):
        ov.setdefault("tet", {})["enabled"] = False
    return ov


def cmd_mesh(a):
    from .process import process_part, summary_row, write_summary
    cfg = load_config(a.config, _overrides(a))
    files = _collect(a.inputs, a.recursive)
    if not files:
        print("aucun fichier STEP", file=sys.stderr)
        return 2
    out = Path(a.output)
    rows = []
    for f in files:
        rep = process_part(f, out, cfg, a.reference_skin, keep_work=a.keep_work)
        row = summary_row(rep)
        rows.append(row)
        print(f"{row['part']:40s} {row['status']:15s} {row['kind'] or '':9s} "
              f"el={row['n_elements'] or 0:>7}  SJmin={row['sj_min'] if row['sj_min'] is None else round(row['sj_min'], 3)}"
              f"  {row['time_s']} s  {row['message'][:80]}")
    write_summary(rows, out / "summary.csv")
    print(f"\nsynthèse : {out / 'summary.csv'}")
    return 0 if all(r["status"] in ("OK", "OK_APPROX", "OK_TET") for r in rows) else 1


def cmd_inspect(a):
    from .analyze.pipeline import PartAnalysis
    from .strategy.runner import run_isolated
    cfg = load_config(a.config, _overrides(a))
    work = Path(a.output) / ".work" / Path(a.file).stem
    work.mkdir(parents=True, exist_ok=True)
    r = run_isolated("stepmesher.strategy.jobs:prepare_job",
                     dict(step=a.file, workdir=str(work), cfg_data=cfg.to_dict(), reference_skin=a.reference_skin,
                          out_json=str(work / "prepare.json")), work / "prepare.json",
                     timeout=cfg["general"]["part_time_budget_s"])
    if r.get("status") != "ok":
        print("analyse en échec :", r.get("reasons"), file=sys.stderr)
        return 1
    pa = PartAnalysis.load(r["analysis"])
    c = pa.classification
    print(f"pièce          : {Path(a.file).name}")
    print(f"import         : {pa.import_info['final']['n_faces']} faces, nettoyage {pa.import_info['healing']}")
    print(f"dimensions OBB : {' x '.join(f'{d:.1f}' for d in pa.obb_dims)} mm (diagonale {pa.diag:.1f})")
    print(f"nature         : {pa.kind}  (dispersion d'épaisseur {c['dispersion']:.3f})")
    print(f"épaisseur      : médiane {c['t_median']:.3f} mm  [p10 {c['t_p10']:.3f} ; p90 {c['t_p90']:.3f}]")
    print(f"paliers        : {[round(z, 3) for z in pa.thickness_zones]}")
    print(f"peaux          : réf. {len(pa.ref_faces)} faces / opp. {len(pa.opp_faces)} faces / "
          f"chants {len(pa.flank_faces)} ; {pa.reference_reason}")
    print(f"plis           : {len(pa.bends)} " + ", ".join(f"R{b['radius']:.2f}/{b['angle_deg']:.0f}°" for b in pa.bends[:8]))
    print(f"arêtes vives   : {len(pa.sharp_edges)}")
    print(f"trous          : {len(pa.holes_kept)} conservés, {len(pa.holes_filled)} bouchés, "
          f"{len(pa.holes_fill_failed)} bouchage en échec")
    print(f"hash exact     : {pa.exact_hash}")
    print(f"temps          : " + ", ".join(f"{k} {v:.1f}s" for k, v in pa.timings.items()))
    for m in pa.messages:
        print("note           :", m)
    if a.json:
        print(json.dumps(pa.to_json(), indent=1, default=str)[:20000])
    return 0


def cmd_gen(a):
    from .testdata.synth import generate_all
    for name, p in generate_all(a.directory).items():
        print(p)
    return 0


def cmd_validate(a):
    from .io.inp_validator import validate_inp
    v = validate_inp(Path(a.file))
    print(json.dumps(v, indent=1, ensure_ascii=False))
    return 0 if v["ok"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="stepmesher", description="Mailleur STEP -> Abaqus (SC8R pour les pièces minces, repli C3D10/C3D4)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="fichier TOML surchargeant la configuration par défaut")
        sp.add_argument("-o", "--output", default="out", help="dossier de sortie (défaut : out)")
        sp.add_argument("--reference-skin", choices=["inner", "outer"], default=None)
        sp.add_argument("--no-fill-holes", action="store_true", help="ne boucher aucun trou")
        sp.add_argument("--fill-holes", type=float, metavar="MM", help="boucher les trous de diamètre <= MM")
        sp.add_argument("--threads", type=int)
        sp.add_argument("-v", "--verbose", action="store_true")

    sm = sub.add_parser("mesh", help="mailler un ou plusieurs STEP (fichiers ou dossiers)")
    sm.add_argument("inputs", nargs="+")
    common(sm)
    sm.add_argument("-r", "--recursive", action="store_true")
    sm.add_argument("--allow-wedge", type=float, metavar="PCT", help="%% de triangles tolérés (SC6R)")
    sm.add_argument("--strategies", help="liste ordonnée, ex. conform,subdiv")
    sm.add_argument("--keep-work", action="store_true", help="conserver les fichiers intermédiaires")
    sm.add_argument("--tet-order", type=int, choices=[1, 2],
                    help="repli tétra : 2 = C3D10 (défaut), 1 = C3D4")
    sm.add_argument("--no-tet", action="store_true", help="désactiver le repli tétraédrique")
    sm.add_argument("--llm", action="store_true", help="activer le conseiller LLM local (llama-server)")
    sm.add_argument("--no-llm", action="store_true", help="désactiver le conseiller LLM")
    sm.add_argument("--llm-url", metavar="URL", help="adresse de llama-server (implique --llm)")
    sm.set_defaults(fn=cmd_mesh)

    si = sub.add_parser("inspect", help="analyse seule : peaux, épaisseurs, plis, trous, temps")
    si.add_argument("file")
    common(si)
    si.add_argument("--json", action="store_true")
    si.set_defaults(fn=cmd_inspect)

    sg = sub.add_parser("gen-testdata", help="générer les pièces synthétiques de test")
    sg.add_argument("directory")
    sg.set_defaults(fn=cmd_gen)

    sv = sub.add_parser("validate", help="contrôler un .inp")
    sv.add_argument("file")
    sv.set_defaults(fn=cmd_validate)

    sd = sub.add_parser("dump-config", help="afficher la configuration par défaut (TOML)")
    sd.set_defaults(fn=lambda a: print(DEFAULT_PATH.read_text()) or 0)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(a, "verbose", False) else logging.WARNING,
                        format="%(levelname)s %(message)s")
    logging.getLogger("stepmesher").setLevel(logging.INFO)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
