"""Non-régression entre deux campagnes : compare les .json par pièce.

    python tools/compare_runs.py result_echelle result_v2 [--tol-el 0.10]

Code retour 1 si au moins une pièce régresse (statut moins bon, nombre d'éléments hors
tolérance signalé, jacobien min, % hors cibles, régularité ou topologie dégradés).
Topologie (`metrics.topology.n_irregular`) : nœuds intérieurs à 3 ou 5+ quads dans les faces
qui devraient être maillées en rangées régulières (pattes, lanières, plis) ; colonne « i ». Pièce absente = signalée
seulement. Régularité (`metrics.regularity`, absente des campagnes antérieures au 28/09/2026) :
% de petits éléments + % de sauts de taille > 1,5 entre voisins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RANK = {"OK": 3, "OK_APPROX": 3, "OK_TET": 2, "FAILED_QUALITY": 1}


def load_run(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(d).glob("*.json")):
        raw = p.read_bytes()
        try:
            r = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:          # anciens rapports écrits en cp1252
            r = json.loads(raw.decode("cp1252"))
        if not isinstance(r, dict) or "part" not in r:
            continue
        m = r.get("metrics") or {}
        out[r["part"]] = dict(status=r.get("status"), el_type=m.get("element_type") or ("SC8R" if m else None),
                              n_el=m.get("n_elements"), sj=(m.get("scaled_jacobian") or {}).get("min"),
                              soft=(m.get("soft_violations") or {}).get("pct"),
                              small=(m.get("regularity") or {}).get("small_pct"),
                              jump=(m.get("regularity") or {}).get("jump_pct"),
                              reg=(m.get("regularity") or {}).get("penalty"),
                              topo=(m.get("topology") or {}).get("n_irregular"),
                              topo_faces=(m.get("topology") or {}).get("n_faces"),
                              kind=(r.get("analysis") or {}).get("kind"),
                              t=(r.get("timings") or {}).get("total"))
    return out


def compare(base: dict, new: dict, tol_el: float = 0.10, tol_reg: float = 5.0, tol_topo: int = 10) -> list[dict]:
    rows = []
    for part in sorted(set(base) | set(new)):
        b, n = base.get(part), new.get(part)
        flags = []
        if n is None:
            flags.append("absente")
        elif b is None:
            flags.append("nouvelle")
        else:
            rb, rn = RANK.get(b["status"], 0), RANK.get(n["status"], 0)
            if rn < rb:
                flags.append("REGRESSION statut")
            elif rn > rb:
                flags.append("gain statut")
            if b["el_type"] != n["el_type"]:
                flags.append(f"type {b['el_type']}->{n['el_type']}")
            if b["n_el"] and n["n_el"] and abs(n["n_el"] - b["n_el"]) > tol_el * b["n_el"]:
                flags.append(f"n_el {n['n_el'] / b['n_el'] - 1:+.0%}")
            if rn == rb and b["sj"] is not None and n["sj"] is not None and n["sj"] < b["sj"] - 0.02:
                flags.append("REGRESSION jacobien")
            if rn == rb and (n["soft"] or 0) > (b["soft"] or 0) + 0.1:
                flags.append("REGRESSION hors cibles")
            if rn == rb and b.get("reg") is not None and n.get("reg") is not None:
                if n["reg"] > b["reg"] + tol_reg:
                    flags.append(f"REGRESSION régularité {b['reg']:.0f}->{n['reg']:.0f}")
                elif n["reg"] < b["reg"] - tol_reg:
                    flags.append(f"gain régularité {b['reg']:.0f}->{n['reg']:.0f}")
            # étoiles dans des faces qui devraient être en rangées régulières (metrics.topology,
            # absent des campagnes antérieures au 29/09/2026 soir)
            if rn == rb and b.get("topo") is not None and n.get("topo") is not None:
                if n["topo"] > b["topo"] + max(tol_topo, 0.2 * b["topo"]):
                    flags.append(f"REGRESSION topologie {b['topo']}->{n['topo']} nœuds irréguliers")
                elif n["topo"] < b["topo"] - max(tol_topo, 0.2 * b["topo"]):
                    flags.append(f"gain topologie {b['topo']}->{n['topo']}")
        rows.append(dict(part=part, base=b, new=n, flags=flags))
    return rows


def _fmt(r):
    if r is None:
        return "-"
    reg = "" if r.get("reg") is None else f" p{r['small']:4.1f}/s{r['jump']:4.1f}"
    topo = "" if r.get("topo") is None else f" i{r['topo']:>4}"
    return f"{r['status'] or '?':14s} {r['el_type'] or '-':6s} {r['n_el'] or 0:>7}{reg}{topo}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--tol-el", type=float, default=0.10, help="écart relatif toléré sur le nb d'éléments")
    ap.add_argument("--tol-reg", type=float, default=5.0,
                    help="hausse tolérée (points) de %% petits éléments + %% sauts de taille")
    ap.add_argument("--tol-topo", type=int, default=10,
                    help="hausse tolérée du nb de nœuds irréguliers dans les faces attendues régulières "
                         "(au moins 20 %% de la valeur de base)")
    a = ap.parse_args(argv)
    rows = compare(load_run(Path(a.base)), load_run(Path(a.new)), a.tol_el, a.tol_reg, a.tol_topo)
    n_reg = 0
    for r in rows:
        bad = any(f.startswith("REGRESSION") for f in r["flags"])
        n_reg += bad
        print(f"{r['part'][:32]:32s} | {_fmt(r['base'])} | {_fmt(r['new'])} | {', '.join(r['flags'])}")
    print(f"\n{len(rows)} pièces, {n_reg} en régression")
    return 1 if n_reg else 0


if __name__ == "__main__":
    sys.exit(main())
