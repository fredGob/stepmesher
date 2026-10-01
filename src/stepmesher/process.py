"""Traitement complet d'une pièce : analyse -> essais de maillage -> export -> rapport."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from .analyze.family import classify_family
from .analyze.feasibility import assess_feasibility
from .analyze.pipeline import PartAnalysis
from .config import Config
from .io.exports import write_orientation_csv, write_vtu, write_vtu_tet
from .io.inp_validator import validate_inp
from .io.inp_writer import write_inp, write_inp_tet
from .strategy.candidates import deterministic_candidates
from .strategy.recipe import Recipe
from .strategy.runner import run_isolated

log = logging.getLogger("stepmesher")

SUMMARY_FIELDS = ["part", "status", "memory", "kind", "family", "feasibility", "feasibility_score", "t_median_mm",
                  "n_zones", "reference_skin", "recipe", "element_type", "n_elements", "sj_min", "sj_p05",
                  "soft_violation_pct", "small_pct", "jump_pct", "irregular_faces", "irregular_nodes",
                  "reprojection_max", "thickness_dev_max", "n_attempts", "time_s",
                  "output", "message"]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def _sc8r_passes(step, work, cfg, reference_skin, report, T0, has_microfix, vtag, vlabel):
    """Passes SC8R (A : micro-arêtes au maillage ; B : micro-arêtes supprimées à l'import)
    sur UNE géométrie. Retourne un dict (pa, analysis_json, winner, best, geometry_pass,
    no_skin) ou None si l'analyse échoue. Ne fait ni tétra ni export."""
    cfg_a = Config(cfg.to_dict())
    cfg_a.data["healing"]["small_edge_tol_mm"] = []
    pa, r = _prepare(step, work, cfg_a, reference_skin, report, T0, key=f"analysis_{vtag or 'A'}")
    if pa is None:
        return None
    _scale_budget(cfg, pa, report)
    out = dict(pa=pa, analysis_json=r["analysis"], winner=None, best=None,
               geometry_pass=f"A ({vlabel})", label=vlabel, no_skin=False, tet_direct=None)
    fam = _analysis_summary(pa, cfg)["family"]
    if fam in cfg["family"].get("tet_direct", []) and cfg["tet"]["enabled"]:
        out["tet_direct"] = fam
        return out
    if not pa.ref_faces or not pa.opp_faces or pa.kind == "massive":
        # faces non triangulées à l'analyse -> « massive » factice (part_024) : retenter en passe B
        if pa.failed_faces and has_microfix:
            log.warning("analyse %s : %d faces en échec, pas de peaux -> passe B (micro-arêtes supprimées)",
                        vlabel, len(pa.failed_faces))
            wb = work / "microfix"
            pa_b, r_b = _prepare(step, wb, cfg, reference_skin, report, T0, key=f"analysis_B_{vtag or 'A'}")
            if pa_b is not None and pa_b.ref_faces and pa_b.opp_faces and pa_b.kind != "massive":
                out.update(pa=pa_b, analysis_json=r_b["analysis"],
                           geometry_pass=f"B ({vlabel}, micro-arêtes supprimées)")
                out["winner"], out["best"] = _sc8r_pass(pa_b, r_b["analysis"], wb, cfg, report, T0,
                                                        tag=f"{vtag}B")
                return out
        out["no_skin"] = True
        return out
    winner, best = _sc8r_pass(pa, r["analysis"], work, cfg, report, T0, tag=f"{vtag}A")
    if winner is None and has_microfix and not report.get("t_junction_early_stop"):
        log.warning("aucune recette SC8R sur la géométrie %s -> passe B (micro-arêtes supprimées à l'import)", vlabel)
        wb = work / "microfix"
        pa_b, r_b = _prepare(step, wb, cfg, reference_skin, report, T0, key=f"analysis_B_{vtag or 'A'}")
        if pa_b is not None and str(pa_b.import_info.get("healing", "")).startswith("micro_edges") \
                and pa_b.ref_faces and pa_b.opp_faces and pa_b.kind != "massive":
            w_b, b_b = _sc8r_pass(pa_b, r_b["analysis"], wb, cfg, report, T0, tag=f"{vtag}B")
            if w_b is not None or (b_b is not None and (best is None or
                                    b_b[1]["metrics"]["score"] > best[1]["metrics"]["score"])):
                out["pa"], out["analysis_json"] = pa_b, r_b["analysis"]
                out["geometry_pass"] = f"B ({vlabel}, micro-arêtes supprimées)"
                winner, best = w_b, b_b
    out["winner"], out["best"] = winner, best
    return out


def _scale_budget(cfg: Config, pa, report: dict) -> None:
    """Grandes pièces : délai par essai et budget par pièce proportionnels au nombre
    d'éléments estimé (demi-aire / h0²), au-delà de `budget_ref_elements`. Une pièce de
    128 m² (upper part_011, ~640 000 SC8R à 10 mm) ne peut pas être maillée en 300 s."""
    g = cfg["general"]
    ref = float(g.get("budget_ref_elements", 0.0))
    if ref <= 0:
        return
    t = pa.classification.get("t_median") or 1.0
    h0 = cfg.target_size(pa.diag, t)
    area = float((pa.import_info.get("final") or {}).get("area", 0.0))
    n_est = 0.5 * area / max(h0 * h0, 1e-9)
    s = max(1.0, n_est / ref)
    if s > report.get("budget_scale", 1.0):
        report["budget_scale"] = round(s, 2)
        report["n_elements_estimate"] = int(n_est)
        a, b = _limits(cfg, report)
        log.info("grande pièce (~%d éléments estimés) : délai par essai %.0f s, budget %.0f s", n_est, a, b)


def _limits(cfg: Config, report: dict) -> tuple[float, float]:
    """(délai par essai, budget par pièce) après mise à l'échelle (_scale_budget)."""
    g = cfg["general"]
    s = float(report.get("budget_scale", 1.0))
    budget = min(g["part_time_budget_s"] * s, max(g.get("part_time_budget_max_s", 0.0), g["part_time_budget_s"]))
    # part du budget réservée aux variantes de géométrie suivantes (brute, nettoyage profond)
    budget *= float(report.get("variant_budget_frac", 1.0))
    return (min(g["attempt_timeout_s"] * s, max(g.get("attempt_timeout_max_s", 0.0), g["attempt_timeout_s"])),
            budget)


def _variant_is_perfect(v, report: dict | None = None) -> bool:
    """Variante SC8R gagnante sans aucun élément hors cibles : inutile d'en essayer une autre.
    Très grande pièce (budget x5 et plus : ~15 min par essai, upper part_011) : une variante
    qui passe suffit."""
    w = v.get("winner")
    if w is None:
        return False
    if report is not None and float(report.get("budget_scale", 1.0)) >= 5.0:
        return True
    return int((w[1]["metrics"].get("soft_violations") or {}).get("count", 0)) == 0


def _variant_rank(v):
    """Clé de tri des variantes : une gagnante prime ; entre gagnantes, la plus régulière
    (moins de petits éléments et de sauts de taille, `quality.regularity`), puis meilleur score.
    Le score seul (% hors cibles) favorisait les maillages fins, qui diluent leurs défauts
    (part_004 : recette MeshAdapt à 2 911 éléments préférée à conform×1 à ~1 230)."""
    w = v.get("winner")
    if w is not None:
        m = w[1]["metrics"]
        pen = (m.get("regularity") or {}).get("penalty", 0.0)
        return (2, -round(pen, 1), m.get("score", 0.0), -int((m.get("soft_violations") or {}).get("count", 0)))
    b = v.get("best")
    return (1 if b is not None else 0, 0.0, b[1]["metrics"].get("score", -1e18) if b else -1e18, 0)


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
        has_microfix = bool(cfg["healing"].get("small_edge_tol_mm"))

        # ---------- SC8R sur la géométrie nettoyée (ou brute si OCP n'a rien changé) ----------
        # budget partagé entre variantes : la géométrie nettoyée n'épuise pas tout (part_018 upper :
        # 16 essais en échec sur la nettoyée, plus de temps pour la variante profonde qui passe)
        report["variant_budget_frac"] = 0.5 if cleaned else 1.0
        v_clean = _sc8r_passes(step_c, work, cfg, reference_skin, report, T0, has_microfix,
                               vtag="", vlabel="nettoyée" if cleaned else "brute")
        if v_clean is None:
            return _finish(report, out_dir, T0)
        pa = v_clean["pa"]
        report["analysis"] = _analysis_summary(pa, cfg)
        log.info("famille : %s (%s)", report["analysis"]["family"], report["analysis"]["family_reason"])
        # ---------- avis de faisabilité SC8R (triage ; déviation optionnelle) ----------
        feas = assess_feasibility(pa)
        report["feasibility"] = feas
        log.info("faisabilité SC8R : %s (score %.2f) - %s", feas["verdict"], feas["score"], feas["reason"])
        if v_clean["tet_direct"]:
            log.warning("famille %s -> tétra direct (pas d'essai SC8R)", v_clean["tet_direct"])
            return _tet_fallback(report, pa, v_clean["analysis_json"], work, out_dir, cfg, stem, T0,
                                 reason=f"famille {v_clean['tet_direct']} : tétra direct", t_part=True)
        if v_clean["no_skin"]:
            log.warning("pièce %s : pas de maillage SC8R possible -> repli tétraédrique",
                        "massive" if pa.kind == "massive" else
                        "profilé/lisse (type T)" if pa.kind == "profile" else "sans peaux identifiées")
            return _tet_fallback(report, pa, v_clean["analysis_json"], work, out_dir, cfg, stem, T0,
                                 reason="pièce massive" if pa.kind == "massive" else
                                        "profilé/lisse à section balayée (type T)" if pa.kind == "profile" else
                                        "peaux non identifiées")
        # tôle (deux peaux, épaisseur constante OU variable) : SC8R obligatoire
        sheet = pa.kind in ("constant", "variable")
        skip_below = cfg["mesh"].get("skip_sc8r_below", 0.0)
        if not sheet and skip_below > 0 and feas["score"] < skip_below and cfg["tet"]["enabled"]:
            log.warning("faisabilité SC8R faible (score %.2f < %.2f) -> repli tétra direct",
                        feas["score"], skip_below)
            return _tet_fallback(report, pa, v_clean["analysis_json"], work, out_dir, cfg, stem, T0,
                                 reason=f"faisabilité SC8R faible ({feas['score']:.2f} < {skip_below})")

        # ---------- SC8R sur la géométrie brute (pré-OCP) : OCP peut sur-nettoyer et écraser
        # des éléments ; on garde le meilleur des deux. Seulement si OCP a nettoyé et que la
        # variante nettoyée n'est pas déjà parfaite. ----------
        variants = [v_clean]
        if cleaned and not _variant_is_perfect(v_clean, report) and not report.get("t_junction_early_stop"):
            log.info("OCP a nettoyé la géométrie -> essai SC8R aussi sur la géométrie brute (pré-OCP), meilleur retenu")
            report["variant_budget_frac"] = 0.75
            v_raw = _sc8r_passes(raw_step, work / "raw", cfg, reference_skin, report, T0,
                                 has_microfix, vtag="R", vlabel="brute (pré-OCP)")
            if v_raw is not None and not v_raw["no_skin"]:
                variants.append(v_raw)

        # ---------- nettoyage plus profond (sommets isolés proches non fusionnables par une
        # simple arête voisine) : en course avec les variantes précédentes, jamais forcé. ----------
        report["variant_budget_frac"] = 1.0
        deep_prec = float(cfg["healing"].get("occ_wireframe_precision_mm_deep", 0.0))
        if deep_prec > cfg["healing"]["occ_wireframe_precision_mm"] and not report.get("t_junction_early_stop") \
                and not _variant_is_perfect(max(variants, key=_variant_rank), report):
            step_deep = _clean_step_occ(raw_step, work / "deep", cfg, report,
                                        precision=deep_prec, report_key="occ_clean_deep")
            if str(step_deep) != str(raw_step):
                log.info("nettoyage profond (%.2f mm) accepté -> essai SC8R en course avec les autres géométries",
                        deep_prec)
                v_deep = _sc8r_passes(step_deep, work / "deep", cfg, reference_skin, report, T0,
                                      has_microfix, vtag="D", vlabel=f"nettoyée profond ({deep_prec:g} mm)")
                if v_deep is not None and not v_deep["no_skin"]:
                    variants.append(v_deep)

        chosen_v = max(variants, key=_variant_rank)
        pa = chosen_v["pa"]
        analysis_json = chosen_v["analysis_json"]
        winner, best = chosen_v["winner"], chosen_v["best"]
        report["analysis"] = _analysis_summary(pa, cfg)
        report["geometry_pass"] = chosen_v["geometry_pass"]
        if len(variants) > 1:
            report["geometry_variants"] = [dict(label=v["label"], geometry_pass=v["geometry_pass"],
                                                won=v["winner"] is not None) for v in variants]
        report["timings"]["meshing"] = time.time() - T0 - report["timings"].get("analysis_A", 0.0)

        # dernier recours (O1, Fred) : plancher « 3 éléments par rayon » levé si aucune recette ne
        # passe ; sur un petit rayon il peut écraser un hexa voisin (part_031, pli R5 près d'un trou)
        if winner is None and cfg["mesh"].get("bend_angle_floor", False) and not _t_topology(report):
            log.warning("aucune recette SC8R -> dernier essai sans le plancher de 3 éléments par rayon")
            cfg_nf = Config(cfg.to_dict())
            cfg_nf.data["mesh"]["bend_angle_floor"] = False
            for i, v in enumerate(sorted(variants, key=_variant_rank, reverse=True)):
                wd = work / "no_bend_floor" / str(i)
                wd.mkdir(parents=True, exist_ok=True)
                w_nf, b_nf = _sc8r_pass(v["pa"], v["analysis_json"], wd, cfg_nf, report, T0, tag=f"F{i}")
                # le meilleur échec doit rester lié à la géométrie retenue (ids de faces)
                if w_nf is None and i == 0 and b_nf is not None and (best is None or
                                                           b_nf[1]["metrics"]["score"] > best[1]["metrics"]["score"]):
                    best = b_nf
                if w_nf is not None:
                    winner, best = w_nf, b_nf
                    pa, analysis_json = v["pa"], v["analysis_json"]
                    report["analysis"] = _analysis_summary(pa, cfg)
                    report["geometry_pass"] = v["geometry_pass"] + ", sans plancher 3 él./rayon"
                    report["bend_floor_relaxed"] = True
                    log.warning("SC8R obtenu en levant le plancher de 3 éléments par rayon (petits rayons à 2 éléments)")
                    break

        sheet = pa.kind in ("constant", "variable")
        # exception : pièce en T (jonction de parois) -> tous les essais SC8R aboutis donnent
        # un maillage en plusieurs morceaux disjoints (part_022) : tétra permis
        t_junction = _t_topology(report)
        if winner is None and sheet and t_junction:
            log.warning("tôle : tous les essais SC8R sont en plusieurs morceaux (pièce en T) -> tétra permis")
            sheet = False
        if winner is None and sheet:
            log.warning("tôle : aucune recette SC8R ne passe les critères -> pas de repli tétra "
                        "(SC8R obligatoire), meilleur essai SC8R exporté pour diagnostic")
        if winner is None and cfg["tet"]["enabled"] and not sheet:
            log.warning("aucune recette SC8R ne passe les critères -> repli tétraédrique")
            report["sc8r_best"] = dict(recipe=best[0].label(), reasons=best[1].get("reasons", [])) if best else None
            rep = _tet_fallback(report, pa, analysis_json, work, out_dir, cfg, stem, T0,
                                reason="toutes les recettes SC8R ont échoué"
                                       + (" (pièce en T)" if t_junction else ""), t_part=t_junction)
            if rep["status"] == "OK_TET" or best is None:
                return rep
            log.warning("repli tétraédrique en échec : export du meilleur essai SC8R pour diagnostic")
        chosen = winner or best
        if chosen is None:
            report.update(status="FAILED", reasons=["aucun essai n'a produit de maillage"])
            return _finish(report, out_dir, T0)
        rc, res = chosen
        thickness_kind = pa.classification.get("thickness_kind", pa.kind)
        status = ("OK_APPROX" if thickness_kind == "variable" else "OK") if winner else "FAILED_QUALITY"
        mesh = dict(np.load(res["mesh_file"]))
        suffix = "" if winner else ".FAILED"
        inp = out_dir / f"{stem}{suffix}.inp"
        wi = write_inp(inp, mesh, pa, rc.to_dict(), res["metrics"], status, part_name_override=stem)
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
                      approximate=(thickness_kind == "variable"))
        if thickness_kind == "variable":
            log.warning("épaisseur variable : maillage APPROCHÉ (marches d'épaisseur non alignées, "
                        "%d éléments à cheval) - jalon 3", res["metrics"].get("n_step_elements", 0))
        log.info("résultat : %s, %d éléments, jacobien min %.3f -> %s", status, res["metrics"]["n_elements"],
                 res["metrics"]["scaled_jacobian"]["min"], inp.name)
        topo = res["metrics"].get("topology") or {}
        if topo.get("n_faces"):
            log.warning("maillage IRRÉGULIER : %d face(s) qui devraient être en rangées régulières (4 coins) "
                        "contiennent %d nœud(s) à 3 ou 5+ éléments :", topo["n_faces"], topo["n_irregular"])
            for fd in topo.get("faces", [])[:10]:
                log.warning("   face %d : %d nœuds irréguliers / %d, côtés %s mm, vers %s", fd["face"],
                            fd["n_irregular"], fd["n_interior"], fd.get("sides_mm"), fd["centre"])
        return _finish(report, out_dir, T0)
    finally:
        log.removeHandler(fh)
        fh.close()
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def _clean_step_occ(step: Path, work: Path, cfg: Config, report: dict,
                    precision: float | None = None, report_key: str = "occ_clean") -> Path:
    """Nettoyage géométrique OCP (ShapeFix_Wireframe) : effondre les micro-arêtes en
    préservant le solide. Renvoie le STEP nettoyé si accepté, sinon la géométrie brute."""
    h = cfg["healing"]
    if not h.get("occ_wireframe", False):
        return step
    from .occ.clean import CleanResult, clean_step
    prec = h["occ_wireframe_precision_mm"] if precision is None else precision
    work.mkdir(parents=True, exist_ok=True)
    kw = dict(precision=prec, max_volume_change=h["occ_wireframe_max_volume_change"],
              fillet_max_arc=float(h.get("occ_chant_fillet_max_arc_mm", 0.0)),
              facet_max_width=float(h.get("occ_chant_facet_max_width_mm", 0.0)),
              facet_time_s=float(h.get("occ_chant_facet_time_s", 60.0)))
    res = None
    if kw["facet_max_width"] > 0 or kw["fillet_max_arc"] > 0:
        # suppression des congés / facettes de chant en sous-processus, avec délai : un seul
        # appel OCC peut durer plusieurs minutes (part_018 upper : 334 s) -> sans eux sinon
        t_lim = float(h.get("occ_chant_timeout_s", 180.0))
        r = run_isolated("stepmesher.occ.clean:clean_step_job",
                         dict(src=str(step), out_path=str(work / "cleaned.step"),
                              out_json=str(work / "clean.json"), **kw), work / "clean.json", timeout=t_lim)
        if r.get("exec_status") == "ok" and "path" in r:
            res = CleanResult(r["status"], r["path"], r.get("messages", []), r.get("stats", {}))
        else:
            log.warning("nettoyage OCP : suppression des chants abandonnée (%s) -> sans", "; ".join(r.get("reasons", []))[:80])
            kw.update(facet_max_width=0.0, fillet_max_arc=0.0)
    if res is None:
        res = clean_step(step, work / "cleaned.step", **kw)
    report[report_key] = dict(status=res.status, messages=res.messages, stats=res.stats, precision_mm=prec)
    for m in res.messages:
        log.info("nettoyage OCP (%.2f mm) : %s", prec, m)
    return Path(res.path)


def _prepare(step, work, cfg, reference_skin, report, T0, key):
    """Analyse en sous-processus isolé. Retourne (PartAnalysis, résultat) ou (None, résultat)."""
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    left = _limits(cfg, report)[1] - (t0 - T0)
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
    prof = pa.classification.get("profile")
    if prof and prof.get("is_profile"):
        log.info("profilé/lisse à section balayée détecté (élongation %.1f, dispersion section %.1f %%) "
                 "-> essais SC8R peau réf/opp conservés", prof["elongation"],
                 100 * prof["section_dispersion"])
    for m in pa.messages:
        log.warning(m)
    return pa, r


def _neighbors(pa) -> dict[int, set[int]]:
    nb: dict[int, set[int]] = {}
    for a, b, _ in pa.face_adjacency:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def _recipe_key(rc) -> tuple:
    return (rc.strategy, rc.size_mult, rc.structured, tuple(sorted(rc.free_faces)), tuple(sorted(rc.face_alg)),
            tuple(sorted(rc.merge)))


def _t_topology(report, n_min: int = 1) -> bool:
    """Pièce en T : au moins n_min essais SC8R maillés, TOUS en plusieurs morceaux disjoints
    (un essai sans maillage, face non maillée, ne dit rien sur la topologie)."""
    meshed = [a for a in report["attempts"] if a.get("status") == "failed" and a.get("n_elements")
              and not a["label"].startswith("tet")]
    return len(meshed) >= max(n_min, 1) and all(any("morceaux disjoints" in r for r in a.get("reasons", []))
                                                 for a in meshed)


def _sc8r_pass(pa, analysis_json, work, cfg, report, T0, tag: str):
    """Essais SC8R sur une géométrie analysée.

    Après un échec, montée gloutonne déterministe : un seul levier appliqué à la meilleure
    recette obtenue jusque-là. Renvoie `(winner, best)`.
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
    k = 0
    while queue and k < g["max_attempts"] + int(cfg["mesh"].get("adaptive_attempts", 4)):
        rc = queue.pop(0)
        key = _recipe_key(rc)
        if key in seen:
            continue
        seen.add(key)
        elapsed = time.time() - T0
        t_attempt, t_budget = _limits(cfg, report)
        if elapsed > t_budget:
            log.warning("budget de temps par pièce épuisé")
            break
        timeout = min(t_attempt, t_budget - elapsed)
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
        n_stop = int(cfg["mesh"].get("t_junction_stop_after", 0))
        if n_stop and cfg["tet"]["enabled"] and _t_topology(report, n_stop):
            log.warning("%d essais SC8R en plusieurs morceaux disjoints : pièce en T -> arrêt des essais SC8R", n_stop)
            report["t_junction_early_stop"] = True
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
                            free = set(base_rc.free_faces) | (set(base_res["implicated_structured"]) & (bend_faces | set((base_res.get("strategy_info") or {}).get("structured_strips", []))))
                            if free == set(base_rc.free_faces) or free == bend_faces:
                                continue
                            d["free_faces"] = tuple(sorted(free))
                        adapt_left[0] -= 1
                        queue.insert(0, Recipe(**d).clamp())
                        break
    return winner, best


def _tet_fallback(report, pa, analysis_json, work, out_dir, cfg, stem, T0, reason: str,
                  t_part: bool = False) -> dict:
    """Essais tétraédriques, chacun en sous-processus isolé (gmsh peut planter) :
    sources géométriques x combinaisons d'algorithmes, arrêt au premier qui passe."""
    from .mesh.tet import TET_ALGOS, tet_sources
    report["variant_budget_frac"] = 1.0          # tout le budget restant pour le tétra
    if t_part and "min_shape_quality_t" in cfg["tet"]:
        cfg = Config(cfg.to_dict())
        cfg.data["tet"]["min_shape_quality"] = float(cfg["tet"]["min_shape_quality_t"])
    t0 = time.time()                    # sert seulement à mesurer la durée de cette phase
    res, best = {}, None
    # sources supplémentaires issues du STEP d'entrée (avant le nettoyage OCP principal à
    # 0,1 mm) : nettoyé à 0,05 mm sans chants, puis tel quel
    extra_steps = []
    raw_step = str(report.get("source", ""))
    if raw_step and raw_step != str(pa.source):
        h = cfg["healing"]
        if h.get("occ_wireframe", False):
            try:
                from .occ.clean import clean_step
                r05 = clean_step(raw_step, work / "tet_occ005.step", precision=0.05,
                                 max_volume_change=h["occ_wireframe_max_volume_change"])
                if r05.status == "cleaned":
                    extra_steps.append(r05.path)
            except Exception as e:  # noqa: BLE001
                log.warning("tétra : nettoyage OCP 0,05 mm en échec (%s)", type(e).__name__)
        extra_steps.append(raw_step)
    n_src = len(tet_sources(pa, extra_steps))

    def _try(si, ai, mult=1.0):
        t_attempt, t_budget = _limits(cfg, report)
        left = t_budget - (time.time() - T0)
        if left < 30:
            return None
        prefix = work / f"tet_{si}{ai}{'' if mult == 1.0 else f'_x{mult:g}'}"
        r = run_isolated("stepmesher.mesh.tet:run_tet_attempt",
                         dict(analysis_json=analysis_json, cfg_data=cfg.to_dict(), out_prefix=str(prefix),
                              source=si, algo=ai, size_mult=mult, extra_steps=extra_steps),
                         prefix.with_suffix(".json"), timeout=min(t_attempt, left))
        r["_src"] = (si, ai)
        m = r.get("metrics", {})
        alg = r.get("algorithms", {})
        report["attempts"].append(dict(index=len(report["attempts"]),
                                       label=f"tet {alg.get('geometry', si)} 2D={alg.get('alg2d')}/3D={alg.get('alg3d')}"
                                             + ("" if mult == 1.0 else f" x{mult:g}"),
                                       status=r.get("status"), exec=r.get("exec_status"),
                                       reasons=r.get("reasons", []), seconds=r.get("timings", {}).get("total"),
                                       n_elements=m.get("n_elements")))
        log.info("essai tétra %s 2D=%s/3D=%s x%g -> %s %s", alg.get("geometry", si), alg.get("alg2d"),
                 alg.get("alg3d"), mult, r.get("status"), "; ".join(r.get("reasons", [])[:1]))
        return r

    def _better(r, b):
        return b is None or r.get("metrics", {}).get("score", -1e9) > b.get("metrics", {}).get("score", -1e9)

    for si in range(n_src):
        for ai in range(len(TET_ALGOS)):
            r = _try(si, ai)
            if r is None:
                break
            if r.get("mesh_file"):
                if _better(r, best):
                    best = r
                break           # maillage obtenu (bon ou non) : source géométrique suivante
        if best is not None and best.get("passed"):
            break
    # au plus 2 réductions de taille, sur la meilleure source seulement
    for mult in list(cfg["tet"].get("retry_size_mults", []))[:2]:
        if best is None or best.get("passed"):
            break
        r = _try(*best["_src"], mult=float(mult))
        if r is not None and r.get("mesh_file") and (r.get("passed") or _better(r, best)):
            best = r
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
    write_inp_tet(inp, mesh, pa, m, status, part_name_override=stem)
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


def _analysis_summary(pa: PartAnalysis, cfg: Config) -> dict:
    fam, why = classify_family(pa.obb_dims, float(pa.classification.get("t_median") or 0.0),
                               pa.classification.get("thickness_kind", pa.kind),
                               [b["angle_deg"] for b in pa.bends], cfg["family"])
    return dict(kind=pa.kind, family=fam, family_reason=why, classification=pa.classification,
                exact_hash=pa.exact_hash,
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
        json.dumps(report, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return report


def summary_row(rep: dict) -> dict:
    m = rep.get("metrics") or {}
    a = rep.get("analysis") or {}
    c = a.get("classification") or {}
    feas = rep.get("feasibility") or {}
    return dict(part=rep["part"], status=rep.get("status"), memory=rep["memory"]["status"], kind=a.get("kind"),
                family=a.get("family"),
                feasibility=feas.get("verdict"), feasibility_score=feas.get("score"),
                t_median_mm=c.get("t_median"), n_zones=len(a.get("thickness_zones") or []),
                reference_skin=a.get("reference_reason"), recipe=rep.get("recipe_label"),
                element_type=m.get("element_type") or ("SC8R" if m else None),
                n_elements=m.get("n_elements"), sj_min=(m.get("scaled_jacobian") or {}).get("min"),
                sj_p05=(m.get("scaled_jacobian") or {}).get("p05"),
                soft_violation_pct=(m.get("soft_violations") or {}).get("pct"),
                small_pct=(m.get("regularity") or {}).get("small_pct"),
                jump_pct=(m.get("regularity") or {}).get("jump_pct"),
                irregular_faces=(m.get("topology") or {}).get("n_faces"),
                irregular_nodes=(m.get("topology") or {}).get("n_irregular"),
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
