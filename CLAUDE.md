# CLAUDE.md — stepmesher

Mailleur Python 100 % local (gmsh 4.15 OCC + numpy + scipy, **aucune API cloud**) :
STEP CATIA V5 (pièces A350 individuelles) → maillage Abaqus, avec contrôle qualité et export
`.inp`. **Nom neutre voulu par Fred** : SC8R (1 élément dans l'épaisseur) est le cas
majoritaire, pas le seul — repli C3D10/C3D4, autres types d'éléments possibles plus tard.
Ancien nom : `step2sc8r` (renommé en `stepmesher`).
Utilisateur : Fred, ingénieur simulation (Airbus), francophone. Répondre en français, par
étapes courtes, poser des questions avant chaque nouveau chantier.

## Commandes

```bash
pip install -e .                                   # + apt: libglu1-mesa libxft2 libxinerama1 libxcursor1 libfontconfig1 libxrender1
stepmesher mesh stps_test/ -o out/                  # lot ; summary.csv + par pièce .inp/.vtu/.json/.log/_orientation.csv
stepmesher inspect piece.stp                        # analyse seule
stepmesher validate out/piece.inp                   # validateur interne
stepmesher dump-config                              # config par défaut (src/stepmesher/default.toml)
pytest -m "not real"                               # 73 tests synthétiques, ~4-5 min
pytest -m real                                     # 7 tests sur stps_test/, ~20 min
```

Environnement de dev utilisé : 2 cœurs / 7 Go → les temps sont pessimistes.

## État (sept. 2026)

Les 5 pièces de `stps_test/` passent :

| Pièce | Statut | Éléments | Remarque |
|---|---|---|---|
| part_000 | OK | ~4 500 SC8R | tôle constante 4,34 mm, 3 plis, 5 s |
| part_093 | OK_TET | ~210 000 C3D10 | jonction en T → SC8R impossible, repli tétra, ~6 min |
| part_145 | OK_APPROX | ~17 300 SC8R | 6 paliers, recette adaptative (5 faces remaillées) |
| part_201 | OK_APPROX | ~12 800 SC8R | 3 paliers, 1 280 BSpline, premier essai |
| part_349 | OK_APPROX | ~67 300 SC8R | 3 paliers, 36 faces remaillées, ~5 min |

Décisions validées avec Fred :
- tolérance « soft » **0,2 %** d'éléments hors cibles (critères durs jamais relâchés) ;
- priorité = **nettoyer le STEP** (surfaces utiles, micro-arêtes, plaque ou non) puis mailler ;
- épaisseur variable (pièces composites) : **transition lissée acceptée** (OK_APPROX), pas
  besoin de marches alignées ; **ne pas traiter le composite** (drapages, 0°, face moule) pour
  l'instant — maillage uniquement ;
- sortie demandée : une `*SURFACE` par peau externe → `SURF_INNER` / `SURF_OUTER`
  (+ alias `SURF_REF` S1 / `SURF_OPP` S2). Intérieure = aire la plus faible ; écart < 0,5 %
  → signalé « quasi plane » (part_201, part_349). Question ouverte : autre règle pour ces cas ?

## Architecture (`src/stepmesher/`)

Pipeline d'une pièce (`process.py::process_part`) :
1. **Passe A, géométrie brute** : `jobs.prepare_job` → `analyze/pipeline.prepare_part`
   (import `occ/loader`, analyse, bouchage de trous `occ/holes`, brep réécrit).
   Massive / sans peau → tétra direct.
2. `_sc8r_pass` : essais isolés (`strategy/runner.run_isolated`, multiprocessing *spawn*,
   timeout) de `strategy/attempt.run_attempt` ; recherche **gloutonne** : un seul levier par
   essai sur la meilleure recette (`alg` → `free` → `merge`), puis tailles et `subdiv`.
3. **Passe B** (workdir `microfix/`) si A échoue et si l'import a accepté la suppression des
   micro-arêtes (`Geometry.OCCFixSmallEdges`).
4. `_tet_fallback` : sources (brep préparé, STEP micro-fix 0,01 / 0,005, STEP brut) ×
   `TET_ALGOS`, chaque combinaison isolée (gmsh segfaulte parfois).
5. Export `io/inp_writer` (+ `write_inp_tet`), `io/exports` (.vtu, CSV), validation
   `io/inp_validator`, rapport JSON.

Modules clés :
- `analyze/skins.py` : rayons vectorisés (KDTree par classes de taille + Möller-Trumbore)
  → épaisseur et face opposée par face ; `exit_cos` filtre les surfaces de *sortie*
  (normale·d > cos). Classement peau/chant par face + propagation, 2-coloration.
- `analyze/classify.py` : constante / variable / massive, paliers par histogramme log pondéré,
  choix de la peau de référence (variable → la moins découpée).
- `mesh/quad.py` : plis transfinis (`_bend_sides`, `_ordered_loop` — **getCurveLoops /
  getBoundary ne sont pas ordonnés**), `h_gen` commun aux plis voisins, `apply_strategy`.
- `mesh/repair.py` : `flip_repair` (bascule + lissage, critère = pire quad remplacé sans
  dégrader les voisins), `fix_micro_edges` (glissement de nœuds).
- `mesh/offset.py` : normales CAD moyennées, onglet `1/cos(θ/2)`, rayons vers peau opposée +
  chants, `nearest_cad` (projection CAD exacte), nœuds sans cible = déplacement des voisins,
  `_untangle`.
- `mesh/quality.py` : critères durs / cibles, exemption « imposé par la CAD »
  (bord < `micro_edge_ratio` × `min_size_mm`, jamais pour les retournés), score.
- `mesh/tet.py` : C3D10 ordre Abaqus retrouvé par géométrie, nœuds milieux droits si
  éléments courbes invalides, `HighOrderOptimize=0` (PETSc absent).

Config : `default.toml`, surcharge stricte (clé inconnue = erreur). **Changer la config
pendant un lot fait planter les sous-processus** (ils relisent le fichier).

## Pièges connus

- Ne jamais `pkill -f <motif>` si le motif figure dans la commande courante (tue le shell).
  Faire `pgrep ... > pids` puis `kill $(cat pids)` dans une commande séparée.
- Scripts utilisant `run_isolated` : garde `if __name__ == "__main__":` obligatoire (spawn).
- Supprimer les micro-arêtes à l'import casse certaines BSpline (face non maillée,
  surfaces auto-intersectantes pour le tétra) → géométrie brute d'abord.
- La stratégie `compound` globale plante / dépasse les délais sur CATIA : retirée des défauts.
- `threads > 1` : gmsh non déterministe (± quelques dizaines d'éléments).
- Les `.inp` sont écrits en ASCII pur (`to_ascii`) : pas d'accents dans Abaqus.
- `summary.csv` : `time_s` du repli tétra corrigé (le budget tétra ne masque plus T0).

## Points à confirmer au premier datacheck Abaqus

1. syntaxe `*DISTRIBUTION` / `*ORIENTATION` par élément ;
2. épaisseur de `*SHELL SECTION` (nominale par zone) vs épaisseur nodale des SC8R ;
3. `STACK DIRECTION=3` avec numérotation 1-4 / 5-8 ;
4. faces S1/S2 des surfaces ; ordre des nœuds milieux C3D10.

## Pistes futures (par priorité proposée)

1. **Campagne sur plus de pièces** (Fred teste) : collecter les `.json` d'échec, classer les
   causes (face non maillée, retournés aux raccords, reprojection, délais) avant de coder.
2. **Retour datacheck Abaqus** : corriger le writer selon les 4 points ci-dessus.
3. **Mémoire de recettes SQLite** (jalon 2) : l'empreinte (hash exact invariant + vecteur de
   caractéristiques) est déjà dans le JSON ; statuts `EXACT_HIT` / `SIMILAR_HIT` /
   `CACHE_FAILED` / `NEW` ; rejouer d'abord la recette gagnante (faces remaillées incluses,
   à rattacher à des ids de faces stables).
4. **Performance** : paralléliser les essais (plusieurs recettes en parallèle), réutiliser
   la géométrie préparée entre passes, réduire les 3-5 min/essai sur part_349.
5. **Règle intérieur/extérieur** pour pièces quasi planes (direction de référence ?).
6. **part_093 / jonctions en T** : découpage en sous-domaines SC8R + raccord (tie) au lieu
   du tout-tétra ; ou mixte SC8R + C3D10 local.
7. **Zones d'épaisseur alignées** (marches nettes) : seulement si besoin de contraintes en
   pied de poche — Fred préfère aujourd'hui la transition lissée.
8. **Composite** (plus tard, sur demande) : correspondance paliers ↔ drapages,
   `*SHELL SECTION, COMPOSITE`, direction 0°, face moule.
9. **LLM local optionnel** (llama.cpp / llama-server, JSON contraint, jamais bloquant) pour
   proposer des leviers de recette à partir des raisons d'échec.
10. Surfaces de chants (`SURF_EDGES`) et sets par bord libre si utiles au chargement.
