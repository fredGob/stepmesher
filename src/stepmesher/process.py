"""Traitement complet d'une pièce : analyse -> essais de maillage -> export -> rapport."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from .analyze.feasibility import assess_feasibility
from .analyze.pipeline import PartAnalysis
from .config import Config
from .io.exports import write_orientation_csv, write_vtu, write_vtu_tet
from .io.inp_validator import validate_inp
from .io.inp_writer import write_inp, write_inp_tet
from .llm.advisor import Advisor
from .llm.payload import build_state
from .strategy.candidates import deterministic_candidates
from .strategy.levers import apply_lever
from .strategy.recipe import Recipe
from .strategy.runner import run_isolated

log = logging.getLogger("stepmesher")

SUMMARY_FIELDS = ["part", "status", "memory", "kind", "feasibility", "feasibility_score", "t_median_mm",
                  "n_zones", "reference_skin", "recipe", "element_type", "n_elements", "sj_min", "sj_p05",
                  "soft_violation_pct", "reprojection_max", "thickness_dev_max", "n_attempts", "time_s",
                  "output", "message"]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def _sc8r_passes(step, work, cfg, reference_skin, report, T0, advisor, has_microfix, vtag, vlabel):
    """Passes SC8R (A : micro-arêtes au maillage ; B : micro-arêtes supprimées à l'import)
    sur UNE géométrie. Retourne un dict (pa, analysis_json, winner, best, geometry_pass,
    no_skin) ou None si l'analyse échoue. Ne fait ni tétra ni export."""
    cfg_a = Config(cfg.to_dict())
    cfg_a.data["healing"]["small_edge_tol_mm"] = []
    pa, r = _prepare(step, work, cfg_a, reference_skin, report, T0, key=f"analysis_{vtag or 'A'}")
    if pa is None:
        return None
    out = dict(pa=pa, analysis_json=r["analysis"], winner=None, best=None,
               geometry_pass=f"A ({vlabel})", label=vlabel, no_skin=False)
    if not pa.ref_faces or not pa.opp_faces or pa.kind == "massive":
        out["no_skin"] = True
        return out
    winner, best = _sc8r_pass(pa, r["analysis"], work, cfg, report, T0, tag=f"{vtag}A",
                              advisor=advisor, allow_microfix=has_microfix)
    if winner is None and has_microfix:
        log.warning("aucune recette SC8R sur la géométrie %s -> passe B (micro-arêtes supprimées à l'import)", vlabel)
        wb = work / "microfix"
        pa_b, r_b = _prepare(step, wb, cfg, reference_skin, report, T0, key=f"analysis_B_{vtag or 'A'}")
        if pa_b is not None and str(pa_b.import_info.get("healing", "")).startswith("micro_edges") \
                and pa_b.ref_faces and pa_b.opp_faces and pa_b.kind != "massive":
            w_b, b_b = _sc8r_pass(pa_b, r_b["analysis"], wb, cfg, report, T0, tag=f"{vtag}B",
                                   advisor=advisor, allow_microfix=False)
            if w_b is not None or (b_b is not None and (best is None or
                                    b_b[1]["metrics"]["score"] > best[1]["metrics"]["score"])):
                out["pa"], out["analysis_json"] = pa_b, r_b["analysis"]
                out["geometry_pass"] = f"B ({vlabel}, micro-arêtes supprimées)"
                winner, best = w_b, b_b
    out["winner"], out["best"] = winner, best
    return out


def _variant_is_perfect(v) -> bool:
    """Variante SC8R gagnante sans aucun élément hors cibles : inutile d'en essayer une autre."""
    w = v.get("winner")
    if w is None:
        return False
    return int((w[1]["metrics"].get("soft_violations") or {}).get("count", 0)) == 0


def _variant_rank(v):
    """Clé de tri des variantes : une gagnante prime, puis meilleur score, puis moins d'éléments hors cibles."""
    w = v.get("winner")
    if w is not None:
        m = w[1]["metrics"]
        return (2, m.get("score", 0.0), -int((m.get("soft_violations") or {}).get("count", 0)))
    b = v.get("best")
    return (1 if b is not None else 0, b[1]["metrics"].get("score", -1e18) if b else -1e18, 0)


def process_part(step: Path, out_dir: Path, cfg: Config, reference_skin: str | None = None,
                 keep_work: bool = False) -> dict:
    step, out_dir = Path(step), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = step.stem
    work = out_dir / ".work" / stem
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    g = cfg["general"]
    T0 = time.time()
    report = dict(part=stem, source=str(step), memory=dict(status="NEW", note="mémoire de recettes : jalon 2"),
                  attempts=[], timings={})
    fh = logging.FileHandler(out_dir / f"{stem}.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    try:
        log.info("=== %s ===", step.name)
        # ---------- nettoyage géométrique OCP (systématique) : micro-arêtes effondrées ----------
        raw_step = step
        step_c = _clean_step_occ(step, work, cfg, report)
        cleaned = str(step_c) != str(raw_step)   # OCP a réellement modifié la géométrie
        advisor = Advisor.from_config(cfg)
        if advisor.enabled:
            log.info("conseiller LLM actif (%s)", cfg["llm"]["base_url"])
        has_microfix = bool(cfg["healing"].get("small_edge_tol_mm"))

        # ---------- SC8R sur la géométrie nettoyée (ou brute si OCP n'a rien changé) ----------
        v_clean = _sc8r_passes(step_c, work, cfg, reference_skin, report, T0, advisor, has_microfix,
                               vtag="", vlabel="nettoyée" if cleaned else "brute")
        if v_clean is None:
            return _finish(report, out_dir, T0)
        pa = v_clean["pa"]
        report["analysis"] = _analysis_summary(pa)
        # ---------- avis de faisabilité SC8R (triage ; déviation optionnelle) ----------
        feas = assess_feasibility(pa)
        report["feasibility"] = feas
        log.info("faisabilité SC8R : %s (score %.2f) - %s", feas["verdict"], feas["score"], feas["reason"])
        if v_clean["no_skin"]:
            log.warning("pièce %s : pas de maillage SC8R possible -> repli tétraédrique",
                        "massive" if pa.kind == "massive" else "sans peaux identifiées")
            return _tet_fallback(report, pa, v_clean["analysis_json"], work, out_dir, cfg, stem, T0,
                                 reason="pièce massive" if pa.kind == "massive" else "peaux non identifiées")
        skip_below = cfg["mesh"].get("skip_sc8r_below", 0.0)
        if skip_below > 0 and feas["score"] < skip_below and cfg["tet"]["enabled"]:
            log.warning("faisabilité SC8R faible (score %.2f < %.2f) -> repli tétra direct",
                        feas["score"], skip_below)
            return _tet_fallback(report, pa, v_clean["analysis_json"], work, out_dir, cfg, stem, T0,
                                 reason=f"faisabilité SC8R faible ({feas['score']:.2f} < {skip_below})")

        # ---------- SC8R sur la géométrie brute (pré-OCP) : OCP peut sur-nettoyer et écraser
        # des éléments ; on garde le meilleur des deux. Seulement si OCP a nettoyé et que la
        # variante nettoyée n'est pas déjà parfaite. ----------
        variants = [v_clean]
        if cleaned and not _variant_is_perfect(v_clean):
            log.info("OCP a nettoyé la géométrie -> essai SC8R aussi sur la géométrie brute (pré-OCP), meilleur retenu")
            v_raw = _sc8r_passes(raw_step, work / "raw", cfg, reference_skin, report, T0, advisor,
                                 has_microfix, vtag="R", vlabel="brute (pré-OCP)")
            if v_raw is not None and not v_raw["no_skin"]:
                variants.append(v_raw)

        chosen_v = max(variants, key=_variant_rank)
        pa = chosen_v["pa"]
        analysis_json = chosen_v["analysis_json"]
        winner, best = chosen_v["winner"], chosen_v["best"]
        report["analysis"] = _analysis_summary(pa)
        report["geometry_pass"] = chosen_v["geometry_pass"]
        if len(variants) > 1:
            report["geometry_variants"] = [dict(label=v["label"], geometry_pass=v["geometry_pass"],
                                                won=v["winner"] is not None) for v in variants]
        report["timings"]["meshing"] = time.time() - T0 - report["timings"].get("analysis_A", 0.0)

        if winner is None and cfg["tet"]["enabled"]:
            log.warning("aucune recette SC8R ne passe les critères -> repli tétraédrique")
            report["sc8r_best"] = dict(recipe=best[0].label(), reasons=best[1].get("reasons", [])) if best else None
            rep = _tet_fallback(report, pa, analysis_json, work, out_dir, cfg, stem, T0,
                                reason="toutes les recettes SC8R ont échoué")
            if rep["status"] == "OK_TET" or best is None:
                return rep
            log.warning("repli tétraédrique en échec : export du meilleur essai SC8R pour diagnostic")
        chosen = winner or best
        if chosen is None:
            report.update(status="FAILED", reasons=["aucun essai n'a produit de maillage"])
            return _finish(report, out_dir, T0)
        rc, res = chosen
        status = ("OK_APPROX" if pa.kind == "variable" else "OK") if winner else "FAILED_QUALITY"
        mesh = dict(np.load(res["mesh_file"]))
        suffix = "" if winner else ".FAILED"
        inp = out_dir / f"{stem}{suffix}.inp"
        wi = write_inp(inp, mesh, pa, rc.to_dict(), res["metrics"], status)
        write_vtu(out_dir / f"{stem}{suffix}.vtu", mesh, wi["zone_of"])
        write_orientation_csv(out_dir / f"{stem}{suffix}_orientation.csv", mesh, wi["zone_names"], wi["zone_of"])
        val = validate_inp(inp)
        if not val["ok"]:
            log.error("validateur .inp : %s", val["errors"][:5])
            if winner:
                status = "INVALID_INP"
        report.update(status=status, recipe=rc.to_dict(), recipe_label=rc.label(), metrics=res["metrics"],
                      reasons=[] if winner else res.get("reasons", []), output=str(inp),
                      sections=wi["zones"], sets=dict(elsets=wi["elsets"], nsets=wi["nsets"], surfaces=wi["surfaces"],
                                skin_areas=wi["skin_areas"]), validation=val,
                      approximate=(pa.kind == "variable"))
        if pa.kind == "variable":
            log.warning("épaisseur variable : maillage APPROCHÉ (marches d'épaisseur non alignées, "
                        "%d éléments à cheval) - jalon 3", res["metrics"].get("n_step_elements", 0))
        log.info("résultat : %s, %d éléments, jacobien min %.3f -> %s", status, res["metrics"]["n_elements"],
                 res["metrics"]["scaled_jacobian"]["min"], inp.name)
        return _finish(report, out_dir, T0)
    finally:
        log.removeHandler(fh)
        fh.close()
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def _clean_step_occ(step: Path, work: Path, cfg: Config, report: dict) -> Path:
    """Nettoyage géométrique OCP (ShapeFix_Wireframe) : effondre les micro-arêtes en
    préservant le solide. Renvoie le STEP nettoyé si accepté, sinon la géométrie brute."""
    h = cfg["healing"]
    if not h.get("occ_wireframe", False):
        return step
    from .occ.clean import clean_step
    res = clean_step(step, work / "cleaned.step",
                     precision=h["occ_wireframe_precision_mm"],
                     max_volume_change=h["occ_wireframe_max_volume_change"])
    report["occ_clean"] = dict(status=res.status, messages=res.messages, stats=res.stats)
    for m in res.messages:
        log.info("nettoyage OCP : %s", m)
    return Path(res.path)


def _prepare(step, work, cfg, reference_skin, report, T0, key):
    """Analyse en sous-processus isolé. Retourne (PartAnalysis, résultat) ou (None, résultat)."""
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    left = cfg["general"]["part_time_budget_s"] - (t0 - T0)
    r = run_isolated("stepmesher.strategy.jobs:prepare_job",
                     dict(step=str(step), workdir=str(work), cfg_data=cfg.to_dict(),
                          reference_skin=reference_skin, out_json=str(work / "prepare.json")),
                     work / "prepare.json", timeout=max(0.0, left))
    report["timings"][key] = time.time() - t0
    if r.get("status") != "ok":
        report.update(status="ANALYSIS_FAILED", reasons=r.get("reasons", []))
        log.error("analyse en échec : %s", r.get("reasons"))
        return None, r
    pa = PartAnalysis.load(r["analysis"])
    log.info("analyse : %s, épaisseur médiane %.3f mm, paliers %s, %d plis, %d trous conservés, "
             "%d trous bouchés (%.1f s)", pa.kind, pa.classification["t_median"],
             [round(z, 2) for z in pa.thickness_zones], len(pa.bends), len(pa.holes_kept),
             len(pa.holes_filled), report["timings"][key])
    for m in pa.messages:
        log.warning(m)
    return pa, r


def _neighbors(pa) -> dict[int, set[int]]:
    nb: dict[int, set[int]] = {}
    for a, b, _ in pa.face_adjacency:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def _sc8r_pass(pa, analysis_json, work, cfg, report, T0, tag: str,
               advisor: "Advisor | None" = None, allow_microfix: bool = False):
    """Essais SC8R sur une géométrie analysée.

    Après un échec, le prochain essai est choisi soit par le conseiller LLM (s'il est
    actif et répond), soit par la montée gloutonne déterministe : un seul levier appliqué
    à la meilleure recette obtenue jusque-là.

    Renvoie `(winner, best)`. Les propositions LLM `microfix` et `tet` sont tracées, mais
    ne peuvent pas écourter les essais SC8R : seul le pipeline détermine le repli.
    """
    g = cfg["general"]
    # T0 = début de la pièce (partagé entre toutes les phases : budget TOTAL par pièce)
    queue = list(deterministic_candidates(cfg))
    seen = set()
    winner, best = None, None
    bend_faces = {f for b in pa.bends for f in b["faces"]}
    tried_alg: set = set()
    history: list = []
    levers_done: set = set()
    adapt_left = [int(cfg["mesh"].get("adaptive_attempts", 4))]
    lc = cfg["llm"]
    k = 0
    while queue and k < g["max_attempts"] + int(cfg["mesh"].get("adaptive_attempts", 4)):
        rc = queue.pop(0)
        key = (rc.strategy, rc.size_mult, rc.structured, tuple(sorted(rc.free_faces)), tuple(sorted(rc.face_alg)),
               tuple(sorted(rc.merge)))
        if key in seen:
            continue
        seen.add(key)
        elapsed = time.time() - T0
        if elapsed > g["part_time_budget_s"]:
            log.warning("budget de temps par pièce épuisé")
            break
        timeout = min(g["attempt_timeout_s"], g["part_time_budget_s"] - elapsed)
        prefix = work / f"attempt_{tag}{k:02d}"
        res = run_isolated("stepmesher.strategy.attempt:run_attempt",
                           dict(analysis_json=analysis_json, recipe=rc.to_dict(), cfg_data=cfg.to_dict(),
                                out_prefix=str(prefix)), prefix.with_suffix(".json"), timeout=timeout)
        m = res.get("metrics", {})
        entry = dict(index=len(report["attempts"]), geometry=tag, recipe=rc.to_dict(), label=rc.label(),
                     status=res.get("status"), exec=res.get("exec_status"), reasons=res.get("reasons", []),
                     seconds=res.get("timings", {}).get("total"), n_elements=m.get("n_elements"),
                     sj_min=(m.get("scaled_jacobian") or {}).get("min"),
                     internal_jacobian_min=(m.get("internal_jacobian") or {}).get("min"))
        report["attempts"].append(entry)
        log.info("essai %s%d %-22s -> %s %s", tag, k, rc.label(), entry["status"],
                 "; ".join(entry["reasons"][:3]) if entry["reasons"] else "")
        k += 1
        if res.get("mesh_file") and (best is None or m.get("score", -9) > best[1].get("metrics", {}).get("score", -9)):
            best = (rc, res)
        if res.get("passed"):
            winner = (rc, res)
            break
        # recette adaptative (montée gloutonne) : chaque essai applique UN levier à la
        # meilleure recette obtenue jusqu'ici ; un levier qui dégrade est abandonné.
        #  "alg"   : faces fautives ou non maillées -> autre algorithme 2D
        #  "free"  : plis structurés au contact d'éléments fautifs -> maillage libre
        #  "merge" : faces étroites fautives -> surface composite avec la voisine tangente
        #            (en dernier : jamais utile sur les pièces CATIA testées)
        if rc.strategy in ("conform", "blossom"):
            sc = m.get("score", -1e9) if res.get("mesh_file") else -1e9
            history.append((sc, rc, res))
            if adapt_left[0] > 0:
                base_sc, base_rc, base_res = max(history, key=lambda h: h[0])
                if base_res.get("bad_faces") is None and base_res.get("faces_not_meshed"):
                    base_res["bad_faces"] = base_res["faces_not_meshed"]
                if base_res.get("bad_faces") is not None:
                    bkey = id(base_rc)
                    # 1) le conseiller LLM choisit le levier, si disponible
                    if advisor is not None and advisor.enabled and \
                            len(history) >= int(lc.get("min_attempts_before", 1)):
                        state = build_state(pa, report["attempts"],
                                            geometry="brute" if tag == "A" else "micro-arêtes supprimées",
                                            max_attempts=int(lc["max_attempts_history"]),
                                            max_faces=int(lc["max_faces"]))
                        dec = advisor.propose(state, base_rc, base_res, bend_faces, tried_alg,
                                              allow_microfix=allow_microfix)
                        report.setdefault("llm", []).append(dict(essai=len(report["attempts"]), **dec.log_entry()))
                        if dec.action in ("tet", "microfix"):
                            # Ces voies sont validées par le pipeline, après les essais SC8R.
                            # Une suggestion LLM ne doit pas écarter une tôle maillable.
                            log.warning("LLM : %s demandé (%s), poursuite des essais SC8R",
                                        dec.action, dec.reason[:120])
                        if dec.recipe is not None:
                            levers_done.add((bkey, dec.action))
                            adapt_left[0] -= 1
                            queue.insert(0, dec.recipe)
                            continue
                    for lever in ("alg", "free", "merge"):
                        if (bkey, lever) in levers_done:
                            continue
                        levers_done.add((bkey, lever))
                        d = base_rc.to_dict()
                        if lever == "merge":
                            new_m = set(map(tuple, base_rc.merge)) | set(map(tuple, base_res.get("sliver_merges", [])))
                            if new_m == set(map(tuple, base_rc.merge)):
                                continue
                            d["merge"] = tuple(sorted(new_m))
                        elif lever == "alg":
                            algs = dict(base_rc.face_alg)
                            ch = {}
                            for f in base_res["bad_faces"]:
                                cur = algs.get(f, 8)
                                nxt = next((a for a in (6, 1, 5) if a != cur and (f, a) not in tried_alg), None)
                                if nxt is not None:
                                    ch[f] = nxt
                            if not ch:
                                continue
                            tried_alg.update(ch.items())
                            algs.update(ch)
                            d["face_alg"] = tuple(sorted(algs.items()))
                        else:
                            if not base_rc.structured or not base_res.get("implicated_structured"):
                                continue
                            free = set(base_rc.free_faces) | (set(base_res["implicated_structured"]) & bend_faces)
                            if free == set(base_rc.free_faces) or free == bend_faces:
                                continue
                            d["free_faces"] = tuple(sorted(free))
                        adapt_left[0] -= 1
                        queue.insert(0, Recipe(**d).clamp())
                        break
    return winner, best


def _tet_fallback(report, pa, analysis_json, work, out_dir, cfg, stem, T0, reason: str) -> dict:
    """Essais tétraédriques, chacun en sous-processus isolé (gmsh peut planter) :
    sources géométriques x combinaisons d'algorithmes, arrêt au premier qui passe."""
    from .mesh.tet import TET_ALGOS, tet_sources
    t0 = time.time()                    # sert seulement à mesurer la durée de cette phase
    res, best = {}, None
    n_src = len(tet_sources(pa))
    for si in range(n_src):
        for ai in range(len(TET_ALGOS)):
            left = cfg["general"]["part_time_budget_s"] - (time.time() - T0)
            if left < 30:
                break
            prefix = work / f"tet_{si}{ai}"
            r = run_isolated("stepmesher.mesh.tet:run_tet_attempt",
                             dict(analysis_json=analysis_json, cfg_data=cfg.to_dict(), out_prefix=str(prefix),
                                  source=si, algo=ai),
                             prefix.with_suffix(".json"), timeout=min(cfg["general"]["attempt_timeout_s"], left))
            m = r.get("metrics", {})
            alg = r.get("algorithms", {})
            report["attempts"].append(dict(index=len(report["attempts"]),
                                           label=f"tet {alg.get('geometry', si)} 2D={alg.get('alg2d')}/3D={alg.get('alg3d')}",
                                           status=r.get("status"), exec=r.get("exec_status"),
                                           reasons=r.get("reasons", []), seconds=r.get("timings", {}).get("total"),
                                           n_elements=m.get("n_elements")))
            log.info("essai tétra %s 2D=%s/3D=%s -> %s %s", alg.get("geometry", si), alg.get("alg2d"),
                     alg.get("alg3d"), r.get("status"), "; ".join(r.get("reasons", [])[:1]))
            if r.get("mesh_file"):
                if best is None or m.get("score", -1e9) > best.get("metrics", {}).get("score", -1e9):
                    best = r
                if r.get("passed"):
                    break
                break           # maillage obtenu mais qualité insuffisante : source géométrique suivante
        if best is not None and best.get("passed"):
            break
    res = best or dict(status="error", reasons=["aucun maillage tétraédrique"], metrics={})
    report["timings"]["tet"] = time.time() - t0
    m = res.get("metrics", {})
    log.info("repli tétra (%s) -> %s %s", reason, res.get("status"), "; ".join(res.get("reasons", [])[:2]))
    if not res.get("mesh_file"):
        report.update(status="FAILED", reasons=[f"repli tétraédrique : {'; '.join(res.get('reasons', []))}"])
        return _finish(report, out_dir, T0)
    mesh = dict(np.load(res["mesh_file"]))
    ok = bool(res.get("passed"))
    status = "OK_TET" if ok else "FAILED_QUALITY"
    inp = out_dir / f"{stem}{'' if ok else '.FAILED'}.inp"
    write_inp_tet(inp, mesh, pa, m, status)
    write_vtu_tet(out_dir / f"{stem}{'' if ok else '.FAILED'}.vtu", mesh)
    val = validate_inp(inp)
    if not val["ok"] and ok:
        status = "INVALID_INP"
        log.error("validateur .inp : %s", val["errors"][:5])
    report.update(status=status, recipe=dict(strategy="tet", order=cfg["tet"]["order"]),
                  recipe_label=f"tet-{m.get('element_type')}", metrics=m, output=str(inp), validation=val,
                  fallback_reason=reason, reasons=[] if ok else res.get("reasons", []))
    log.info("résultat : %s, %s %s -> %s", status, m.get("n_elements"), m.get("element_type"), inp.name)
    return _finish(report, out_dir, T0)


def _analysis_summary(pa: PartAnalysis) -> dict:
    return dict(kind=pa.kind, classification=pa.classification, exact_hash=pa.exact_hash,
                features=pa.features, diag=pa.diag, obb_dims=pa.obb_dims, reference_side=pa.reference_side,
                reference_reason=pa.reference_reason, n_ref_faces=len(pa.ref_faces),
                n_opp_faces=len(pa.opp_faces), n_flank_faces=len(pa.flank_faces),
                failed_faces=pa.failed_faces, thickness_zones=pa.thickness_zones,
                bends=[dict(radius=b["radius"], angle_deg=b["angle_deg"], faces=b["faces"]) for b in pa.bends],
                n_sharp_edges=len(pa.sharp_edges),
                holes_kept=[dict(diameter=h["diameter"], center=h["center"]) for h in pa.holes_kept],
                holes_filled=[dict(diameter=h["diameter"], center=h["center"], axis=h["axis"])
                              for h in pa.holes_filled],
                holes_fill_failed=len(pa.holes_fill_failed), import_info=pa.import_info,
                timings=pa.timings, messages=pa.messages)


def _finish(report: dict, out_dir: Path, T0: float) -> dict:
    report["timings"]["total"] = time.time() - T0
    (out_dir / f"{report['part']}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=_json_default))
    return report


def summary_row(rep: dict) -> dict:
    m = rep.get("metrics") or {}
    a = rep.get("analysis") or {}
    c = a.get("classification") or {}
    feas = rep.get("feasibility") or {}
    return dict(part=rep["part"], status=rep.get("status"), memory=rep["memory"]["status"], kind=a.get("kind"),
                feasibility=feas.get("verdict"), feasibility_score=feas.get("score"),
                t_median_mm=c.get("t_median"), n_zones=len(a.get("thickness_zones") or []),
                reference_skin=a.get("reference_reason"), recipe=rep.get("recipe_label"),
                element_type=m.get("element_type") or ("SC8R" if m else None),
                n_elements=m.get("n_elements"), sj_min=(m.get("scaled_jacobian") or {}).get("min"),
                sj_p05=(m.get("scaled_jacobian") or {}).get("p05"),
                soft_violation_pct=(m.get("soft_violations") or {}).get("pct"),
                reprojection_max=(m.get("reprojection_error") or {}).get("max"),
                thickness_dev_max=(m.get("thickness_deviation") or {}).get("max"),
                n_attempts=len(rep.get("attempts", [])), time_s=round(rep["timings"]["total"], 1),
                output=rep.get("output", ""), message="; ".join(rep.get("reasons", [])[:2]))


def write_summary(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
