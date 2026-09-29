# AGENTS.md — stepmesher

Guide opérationnel pour un agent de code. L'architecture détaillée est dans
[CLAUDE.md](CLAUDE.md) — le lire pour tout travail non trivial. Ce fichier résume
l'essentiel : environnement, commandes, pièges, et le chantier en cours.

## Projet

Mailleur Python 100 % local (gmsh 4.15 OCC + numpy + scipy, **aucune API cloud**) :
STEP CATIA V5 (pièces A350 individuelles) → maillage Abaqus, contrôle qualité, export `.inp`.
Cible majoritaire **SC8R** (1 élément dans l'épaisseur) ; repli C3D10/C3D4.
Utilisateur : **Fred**, ingénieur simulation (Airbus), francophone. **Répondre en français**,
par étapes courtes, poser les questions avant chaque nouveau chantier.

## Environnement (Windows) — À FAIRE EN PREMIER

Le venv de Fred est à `C:\Users\SA516207\WORK\.venv` (le `python`/`py` nu n'a AUCUN paquet).
Avant toute commande Python dans un terminal, l'activer :

```powershell
C:\Users\SA516207\WORK\.venv\Scripts\Activate.ps1
```

- PowerShell : **ne jamais utiliser `&&`** — chaîner avec `;`.
- Terminal PowerShell : artefacts d'encodage sur les accents (é→Ú, è→Þ, ×→Î) dans les
  sorties — cosmétique, ignorer.
- `cadquery-ocp` (kernel OpenCASCADE, module `OCP`) est installé dans ce venv.

## Commandes

```powershell
pip install -e .                 # base
pip install -e .[clean]          # + kernel OCP (cadquery-ocp) : nettoyage géométrique du STEP
pip install -e .[test]           # + pytest

stepmesher mesh <dossier|fichiers.stp> -o out/    # lot ; summary.csv + par pièce .inp/.vtu/.json/.log
stepmesher mesh piece.stp -o out --no-tet         # SC8R uniquement (échoue vite, utile en debug)
stepmesher mesh piece.stp -o out --keep-work      # conserve out/.work/<stem>/ (STEP nettoyé, etc.)
stepmesher inspect piece.stp                       # analyse seule
stepmesher validate out/piece.inp                  # validateur interne
stepmesher dump-config                             # config par défaut (src/stepmesher/default.toml)

pytest -m "not real"             # 75 tests synthétiques (~1-5 min)
pytest -m real                   # tests sur stp_verif/ (lents, ~20 min)
python tools/compare_runs.py result_v7 result_v8   # NON-RÉGRESSION (référence = result_v7, code 1 si régression)
```

Pièces de référence dans `stp_verif/` (part_000, 093, 145, 201, 349). Toutes passent.

## Nettoyage géométrique OCP (feature récente)

Étape 0 du pipeline (`process.py::_clean_step_occ` → `occ/clean.py::clean_step`,
config `[healing] occ_wireframe`) : effondre les micro-arêtes (< `occ_wireframe_precision_mm`,
défaut 0,05 mm) via **`ShapeFix_Wireframe.FixSmallEdges`** (kernel OCP), écrit un STEP nettoyé
repris par toute la suite. **Systématique** (voulu par Fred).

- Rejeté si le nombre de solides change, dV/V > seuil, ou forme invalide → géométrie brute.
- OCP absent → `unavailable`, pipeline continue sur la géométrie brute.
- A débloqué part_201 : `FAILED_QUALITY` (43 min) → `OK_APPROX` 12 054 SC8R (48 s).
- Outils OCC testés et ÉCARTÉS pour ce besoin (ne pas y revenir sans idée neuve) :
  `UnifySameDomain`, `ShapeFix_Wire.FixSmall` (détruit le solide), `Sewing`, `Defeaturing`
  (no-op), `OCCFixSmallEdges` de gmsh (casse les wires). Seul `ShapeFix_Wireframe` marche.

## Pièges connus

- Ne jamais `pkill -f <motif>` / `Stop-Process` si le motif figure dans la commande courante
  (tue le shell). Faire `pgrep ... > pids` puis `kill $(cat pids)` en commande séparée.
- Scripts utilisant `run_isolated` : `if __name__ == "__main__":` obligatoire (multiprocessing spawn).
- **Un STEP sans solide** (compound/shell) casse tout : analyse « massive », épaisseur fausse,
  gmsh « aucun solide ». Le nettoyage doit préserver un `TopoDS_Solid` valide.
- OCP et gmsh embarquent chacun leur OpenCASCADE : ils cohabitent OK dans le même processus
  (testé), mais garder l'import OCP paresseux (dans `clean_step`) ; les sous-processus spawn
  importent `strategy/jobs.py` (pas `process.py`/`occ.clean`) → OCP non chargé côté worker.
- Changer la config pendant un lot fait planter les sous-processus (ils relisent le fichier).
- `threads > 1` : gmsh non déterministe (± quelques dizaines d'éléments, jacobien min variable) :
  défaut `threads = 1` depuis v6 pour une non-régression fiable.
- Les `.inp` sont en ASCII pur (pas d'accents dans Abaqus).
- Nettoyer les fichiers/dossiers temporaires (`proto_*`, sorties de debug) après usage.

## Workflow attendu

1. Lire `CLAUDE.md` (+ la mémoire repo `/memories/repo/`) avant un nouveau chantier.
2. Prototyper isolément (script jetable) avant d'intégrer ; mesurer sur `stp_verif/`.
3. Après toute modif : `pytest -m "not real"` + relancer les pièces réelles impactées
   (non-régression : statuts et nb d'éléments stables).
4. Mettre à jour `CLAUDE.md` et la mémoire repo.

## Chantier en cours / prochain pas

**29/09/2026 après-midi (machine Linux)** : campagnes réelles dans `campagne/` (echelle 44 +
upper 8) ; relaxation du contour, reprojection des nœuds lissés, génératrices des tranches de pli,
facettes de chant, grandes pièces (budget, algo 6, analyse plafonnée). Détail et résultats :
section « 29/09/2026 après-midi » de `CLAUDE.md`. Lancer les campagnes avec
`tools/run_campaign.sh` depuis une copie figée de `src/`.

**⚠ PRIORITAIRE (28/09/2026)** : Fred juge plusieurs maillages de `result_v7` « trop moches »
malgré les critères OK : **trop d'éléments dans les rayons**, sauts de taille brutaux
(part_004, pli R3 105° en 3 faces CAD -> ~9 rangées). Détail et piste : section
« PROCHAINE TÂCHE PRIORITAIRE » de `CLAUDE.md`. Ne rien coder sans en discuter avec Fred.
**Fait (28/09/2026, après accord de Fred)** : pas de champ de taille pour les plis transfinis
(`sizing.apply_sizing(..., structured=)`) + départage des recettes gagnantes par la régularité
(`quality.regularity`, colonnes `small_pct`/`jump_pct`) : 004 2 183 -> 1 230, 014 3 968 -> 1 383.
Campagne complète à relancer et comparer à `result_v7`. Détail : CLAUDE.md.

**Fait (28/09/2026)** : LLM retiré ; `tools/compare_runs.py` (référence = `result_echelle/`,
44 pièces : 31 OK, 1 OK_APPROX, 10 OK_TET, 2 FAILED_QUALITY) ; profilés : `offset`/`inp_writer`
testent `thickness_kind` ; « massive » factice (faces non triangulées, part_024-034 pairs) ->
analyse repassée en passe B avant le tétra ; `.json` écrits en UTF-8 (avant : cp1252).

**Prochain pas — catégorisation par familles** (validé avec Fred) : clips, cornières, lisses,
cadres, ferrures, peaux, traverses (longues, ex. part_036). Axes : échelle, topologie (nb de
plis, jonction T), épaisseur, détails. Une recette (toml) par famille.
Règles de Fred : **taille nominale 10 mm** (grandes pièces), **~8 mm petites pièces** ;
**3 éléments dans les rayons**, pas plus. **Fait** : `h0 = 10·min(1, (diag/600)^0,15)` (continu,
`[mesh] nominal_*`, 130 mm -> 7,9 ; ancienne règle si `nominal_size_mm = 0`) + `bend_angle_floor`
(le plancher `min_bend_size_frac·h0` ne réduit plus un pli sous 3 él./90°). À mesurer sur les 44
pièces contre `result_v2` (campagne de Fred, ancienne règle). Piste suivante : N él. min sur la
largeur d'une aile.
**Fait après campagne v3** : `soft_violation_min_count = 5` (hors cibles tolérés = max(0,2 % N, 5)),
`size_variants = [1.0, 0.7, 0.5]`, tétra `thickness_factor = 2` (ferrure 026 : 16 900 -> 5 500 C3D10),
exception T : seuls les essais ayant produit un maillage comptent (024/030). Reste : 031 (gauchissement
sur peau courbe -> plafonner h par la courbure, C2).
**Fait après campagne v4** : tétra = raffinement local autour des courbes < h/2 (`[tet] short_curve_frac`,
nœuds de congés de 020 : pas un défaut de nettoyage) + au plus 2 réductions de taille
(`retry_size_mults`) ; arrêt précoce SC8R si 2 essais maillés sont tous en morceaux disjoints
(`[mesh] t_junction_stop_after`, pièces en T -> tétra). La fraction de peau (saf) NE sépare PAS
les T (plaque à poches synthétique 0,873 vs T <= 0,86) : écartée. 031 : cause = `bend_angle_floor`
(pli R5 forcé à 4 él. par parité au bord d'un trou Ø14 -> 1 hexa à jacobien 0,08) ; sans le
plancher 031 OK. **O1 validé par Fred et fait** : si aucune recette SC8R ne passe, passe de dernier
recours sans `bend_angle_floor` sur chaque géométrie (`report["bend_floor_relaxed"]`) -> 031 OK.
Reste : 001 (profilé en T 590 mm, 432 faces) : tétra à qualité 0,031-0,045 < 0,05 même à x0,5.
**Fait (v5 : 43/44 OK)** : seuil tétra 0,03 pour les pièces en T (`min_shape_quality_t`, Fred) ;
famille de pièce `analyze/family.py` (`[family]`, règles sur OBB/épaisseur/plis), INFORMATIVE
(rapport `analysis.family` + `summary.csv`). **Vocabulaire et vérité terrain de Fred** (44/44 dans
`tests/test_family.py`) : frame (010, 036), intercostale (021-033 impairs), fitting (019, 020,
022-034 pairs, 035), stabilo (008, 009, 016, 017, 037-043), clip (000, 002-007, 011-015, 018),
clip_autre (001). Prochain pas : une recette par famille.
Ferrures (part_019/020/022/024/035 : SC8R en N morceaux) -> tétra direct probable.
**Fait (v6 : 44/44 OK, 18 min)** : stabilo = clip pour le maillage (Fred) ;
familles `fitting` et `clip_autre` -> tétra direct (`[family] tet_direct`, seuil tétra T 0,03) ;
001 OK_TET (59 700 C3D10). `threads = 1` (reproductibilité). Fixture de test : `tet_direct = []`
(plaque_poches synthétique serait un fitting). Fred : pas d'autre règle par famille pour l'instant.
**RÉFÉRENCE = `result_v7/`** (44/44, statuts identiques à v6, 1er run à `threads = 1`, 20 min).
Fred vérifie les pièces une à une : attendre son retour avant le prochain chantier.

**Ancien chantier — analyser la campagne complète de Fred** :
- collecter les `.json` + `summary.csv`, classer les statuts et les causes d'échec restantes ;
- pour le nettoyage : compter `cleaned` / `unchanged` / `rejected` / `error`, distribution
  du nombre de micro-arêtes effondrées et des dV/V, examiner les rejets ;
- décider si ajuster `occ_wireframe_precision_mm` ou traiter d'autres profils de défauts
  (faces-slivers d'aire faible, micro-faces) — **le defeaturing est un no-op, ne pas y revenir**.
