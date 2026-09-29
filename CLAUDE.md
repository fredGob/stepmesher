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
pip install -e .[clean]                             # + kernel OCP (cadquery-ocp) : nettoyage géométrique du STEP
stepmesher mesh stps_test/ -o out/                  # lot ; summary.csv + par pièce .inp/.vtu/.json/.log/_orientation.csv
stepmesher inspect piece.stp                        # analyse seule
stepmesher validate out/piece.inp                   # validateur interne
stepmesher dump-config                              # config par défaut (src/stepmesher/default.toml)
pytest -m "not real"                               # 85 tests synthétiques, ~4-5 min
pytest -m real                                     # 7 tests sur stps_test/, ~20 min
```

Environnement de dev utilisé : 2 cœurs / 7 Go → les temps sont pessimistes.

## ⚠ PROCHAINE TÂCHE PRIORITAIRE (retour de Fred, 28/09/2026) — qualité visuelle du maillage

La campagne `result_v7` passe 44/44 **les critères**, mais Fred juge plusieurs maillages
**« vraiment trop moches », inacceptables pour un ingénieur calcul**. Problème récurrent
(déjà signalé plusieurs fois) : **trop d'éléments dans les rayons** et **écarts de taille
trop brutaux** entre éléments voisins. Les critères qualité actuels (jacobien, angles,
élancement, gauchissement par élément) ne voient PAS ce défaut : un maillage peut être OK et
mauvais. Règle de Fred rappelée : **3 éléments dans un rayon, pas plus**, taille nominale
~8-10 mm, transitions douces.

Cas de référence : **part_004** (clip, t = 2,17 mm, diag 328 mm, recette `conform×1`,
2 181 SC8R, capture de Fred) :
- le pli R3 à 105° porte ~9 rangées d'éléments fins au lieu de 3. **Hypothèse non vérifiée** :
  ce pli est découpé en **3 faces CAD** (`bends` : faces [32, 33, 34] ; les 2 autres plis R3/90°
  n'ont qu'une face) et `quad.structured_bends` applique la règle 1 él./30° (+ parité)
  **à chaque face** au lieu de répartir 3 éléments sur **le pli entier** ; idem côté
  `sizing.apply_sizing` (taille au pli calculée par pli, mais appliquée face par face ?) ;
- à côté : amas de petits éléments / étoiles le long d'un bord de la peau (face plane
  voisine du pli), et bande fine sur le congé de chant à gauche, avec des sauts de taille
  x3-x5 vers les éléments de 8-10 mm.

**Diagnostic du 28/09/2026 (part_004, part_014)** : l'hypothèse des plis multi-faces est
**fausse** ; les plis transfinis ont bien 3-4 rangées (4 = parité quand l'arc borde une autre
face de peau, accepté par Fred). Les éléments fins sont dans les faces VOISINES :
- le champ Threshold des plis (`sizing.py`) raffinait autour des génératrices jusqu'à ~2 h0
  → rosaces le long des plis, ailes étroites entièrement fines. **Corrigé** : pas de champ pour
  un pli dont toutes les faces sont transfinies (`apply_sizing(..., structured=)`) ;
- la gradation autour des arêtes CAD courtes (4 x h0) est toujours active ; la raccourcir
  (1,5-2,5 x h0) ou lui mettre un plancher (0,15-0,3 x h0) fait échouer 004/014 en élancement
  près des micro-arêtes → recette MeshAdapt, pire. Laissée telle quelle (`short_curve_dist_frac`) ;
- le choix entre variantes qui passent se faisait sur `metrics.score` = -10 x % hors cibles :
  un maillage 3x plus fin dilue ses défauts et gagnait (004 : `RA4` MeshAdapt 2 911 él. retenu
  alors que `A0 conform×1` passait). **Corrigé** : `quality.regularity` (non bloquant, plis
  transfinis exemptés, taille nominale sans size_mult) -> `metrics.regularity` {small_pct : %
  quads < 0,5 h0 ; jump_pct : % arêtes partagées avec rapport de tailles > 1,5 (seuil de Fred) ;
  penalty = somme} ; `process._variant_rank` départage les gagnantes par penalty puis score.
  Colonnes `small_pct`/`jump_pct` dans `summary.csv` ; `compare_runs.py` signale
  « REGRESSION régularité » si penalty +5 points (`--tol-reg`).
- MeshAdapt (algo 1) n'est PAS en cause : il ne sur-raffinait qu'à cause de l'ancien champ pli
  (014 face 2 sans ce champ : 1 383 él., passe). Ne pas le restreindre.
- Résultat : **004 2 183 -> 1 230 SC8R (`conform×1`), 014 3 968 -> 1 383** ; tests 122 OK.
- Reste : (a) le levier « plis libres » réactive le champ pli (normal : pli libre isotrope) ->
  ~4 000 él. sur 014 si ce levier gagne ; piste : Threshold à DistMax plus court ou `Restrict`
  aux faces du pli ; (b) rosaces aux coins arrondis du contour (gradation arêtes courtes 4 x h0) ;
  (c) small_pct reste ~35-40 % même sur un maillage propre (gradations près des plis / arêtes
  courtes) : indicateur relatif, pas un seuil absolu ; (d) **campagne 44 pièces à relancer**
  et comparer à `result_v7` (`compare_runs.py`).

Piste de travail (à discuter avec Fred avant de coder) :
1. diagnostic chiffré sur part_004 puis sur toute `result_v7` : par pli (toutes ses faces),
   nombre d'éléments en travers ; taille min / taille voisine (gradient) par face ;
2. corriger la règle des plis multi-faces (3 éléments sur le pli complet) ;
3. ajouter un critère « esthétique » mesurable dans `mesh/quality.py` (ratio de taille entre
   éléments adjacents, nb d'éléments en travers d'un pli) et le suivre dans `compare_runs.py`,
   pour que la non-régression voie enfin ce défaut ;
4. vérifier visuellement avec Fred sur quelques pièces (004 d'abord).

## État (sept. 2026)

Les 5 pièces de `stps_test/` passent :

| Pièce | Statut | Éléments | Remarque |
|---|---|---|---|
| part_000 | OK | ~4 500 SC8R | tôle constante 4,34 mm, 3 plis, 5 s |
| part_093 | OK_TET | ~210 000 C3D10 | jonction en T → SC8R impossible, repli tétra, ~6 min |
| part_145 | OK_APPROX | ~17 300 SC8R | 6 paliers, recette adaptative (5 faces remaillées) |
| part_201 | OK_APPROX | ~12 000 SC8R | 3 paliers ; **133 micro-arêtes effondrées par nettoyage OCP**, sans quoi FAILED_QUALITY (éléments écrasés) |
| part_349 | OK_APPROX | ~66 700 SC8R | 3 paliers, 36 faces remaillées, ~5 min |

Décisions validées avec Fred :
- tolérance « soft » **0,2 %** d'éléments hors cibles pour les tôles **constantes** ;
  **1,0 %** pour les pièces à épaisseur **variable** (rampes : quelques hexa en biais
  inévitables à la transition lissée) — `[quality] soft_violation_pct{,_variable}` ;
  critères durs jamais relâchés ;
- priorité = **nettoyer le STEP** (surfaces utiles, micro-arêtes, plaque ou non) puis mailler ;
- épaisseur variable (pièces composites) : **transition lissée acceptée** (OK_APPROX), pas
  besoin de marches alignées ; **ne pas traiter le composite** (drapages, 0°, face moule) pour
  l'instant — maillage uniquement ;
- **tôle (kind `constant` OU `variable`) ⇒ SC8R obligatoire** (« pas de concession ») : jamais de repli
  tétra ni de `skip_sc8r_below` ; en échec, meilleur SC8R exporté en FAILED_QUALITY
  (`process.py`, `sheet`). Seule exception : pièce en T = tous les essais SC8R en plusieurs
  morceaux (part_022) -> tétra permis ; `profile` garde le repli tétra ;
- **maillage en plusieurs morceaux disjoints = critère dur** (`quality.mesh_pieces`, SC8R et
  tétra ; cas part_022, peau de référence coupée par les chants des marches) ;
- **plis : 3 éléments sur 90°** (`max_bend_angle_deg = 30`, `n_per_bend = 2`) ; nombre impair
  permis quand l'arc est sur un bord libre (parité seulement si l'arc borde une autre face de
  peau) ; `curvature_n = 3` (pas de sur-raffinement isotrope des congés) ;
- **éléments trop fins sur les faces planes** (sept. 2026) : cause principale = anneaux
  concentriques autour des arêtes CAD courtes (Threshold sur 4 x h0 dans `sizing.py`), qui ne
  servaient qu'à donner >= 2 segments à ces arêtes : **gmsh full-quad (RecombinationAlgorithm
  3) divise par 2 le maillage 1D de CHAQUE courbe d'une face libre** (« 1D mesh cannot be
  divided by 2 » -> face non maillée). Remplacé par `quad.even_boundaries` (maillage 1D, +1
  segment aux courbes impaires, avant le 2D). Les courbes composites de
  `merge_micro_curves` **gardent leur nœud commun** (parité par constituant) : la fusion ne
  supprime pas le nœud imposé. Ordre dans `attempt` : stratégie PUIS champ de taille
  (`apply_sizing(..., skip_curves=courbes fusionnées)`). 021 : 15 % -> 4 % de petits
  éléments ; `[mesh] short_curve_min_size_frac` (plancher, 0 = off) ;
- **lanières** (`quad.structured_strips`, faces longues et étroites <= 4 h0, longueur >= 3 x
  largeur, bouts ~droits <= 1,5 x largeur) en transfini : supprime les « étoiles » (triangle
  orphelin du full-quad coupé en 3 quads). Bout long/découpé (redans, part_029 face 70 ;
  part_021 face 59) = exclu (transfini tordu, éléments retournés) ; piste : sous-découpage.
  Libérables par le levier `free` comme les plis. Aucun réglage gmsh global (algo 5/6/11,
  recombinaison 0/1/2, RecombineOptimizeTopology) ne fait mieux ;
- **micro-congés de chant** (`occ/clean.py::_defeature_chant_fillets`, `[healing]
  occ_chant_fillet_max_arc_mm = 3`) : cylindre de chant (hauteur 0,7-1,5 x épaisseur 2V/A)
  d'arc court supprimé par `BRepAlgoAPI_Defeaturing` (les chants voisins se prolongent) ;
  son arc imposait deux sommets rapprochés sur la peau -> amas au coin du contour (part_025,
  2 % -> 1 %). Sur la campagne : 002-007, 011-015, 019, 021, 023, 025, 027 concernées, tous
  les candidats vérifiés = chants, dV/V <= 3e-4. `UnifySameDomain` n'y fait rien (arêtes
  entre faces différentes). Acceptation propre, indépendante de l'effondrement des arêtes ;
- **tétra : bande régulière dans les congés** (`tet.fillet_bands`, `[tet] fillet_*`) : cylindres
  simples à 4 courbes, arcs égaux ±20 %, transfini 1 él./30° en travers (>= 2 au-delà de 45°) ;
  si le 3D échoue avec bandes -> même essai sans bandes (congé coupé en biais : part_024 face 26) ;
- sortie demandée : une `*SURFACE` par peau externe → `SURF_INNER` / `SURF_OUTER`
  (+ alias `SURF_REF` S1 / `SURF_OPP` S2). Intérieure = aire la plus faible ; écart < 0,5 %
  → signalé « quasi plane » (part_201, part_349). Question ouverte : autre règle pour ces cas ?

## Architecture (`src/stepmesher/`)

Pipeline d'une pièce (`process.py::process_part`) :
0. **Nettoyage OCP** (`occ/clean.py`, `_clean_step_occ`, `[healing] occ_wireframe`) : effondrement
   des micro-arêtes (< `occ_wireframe_precision_mm`, défaut 0,05 mm) via
   `ShapeFix_Wireframe.FixSmallEdges` du kernel OpenCASCADE (module OCP, `cadquery-ocp`),
   écrit un STEP nettoyé repris par toute la suite. Contexte de partage global cohérent —
   là où `ShapeFix_Wire.FixSmall` par contour, `UnifySameDomain`, `Defeaturing`, `Sewing` et
   `OCCFixSmallEdges` de gmsh cassent le solide ou n'ont aucun effet sur les slivers. Rejeté
   si le nombre de solides change, dV/V > `occ_wireframe_max_volume_change`, ou forme invalide
   → géométrie brute conservée. OCP absent → nettoyage désactivé (`unavailable`). OCP et gmsh
   cohabitent dans le même processus ; les sous-processus (spawn) n'importent pas OCP.
   Import OCP **tolérant aux versions** (méthodes statiques `_s`/sans `_s`, `TopExp_Explorer`
   si le map indexé a disparu). **Le nettoyage OCP peut sur-nettoyer** (effondrer des arêtes
   utiles → hexa écrasés, ex. part_001) : quand OCP a réellement nettoyé, le SC8R est tenté
   **sur la géométrie nettoyée ET sur la brute**, meilleur retenu (part_201 profite du nettoyage,
   part_001 récupère son SC8R). Pas de double coût si OCP n'a rien changé.
0bis. **Avis de faisabilité SC8R** (`analyze/feasibility`) : score 0..1 + verdict
   (sc8r / hard / tet) sur les features (couverture de peau, équilibre des peaux, faces en échec),
   journalisé + `summary.csv`/JSON (triage). Déviation directe vers le tétra seulement si
   `[mesh] skip_sc8r_below > 0` (désactivée par défaut, à calibrer sur une campagne).
1. **Passe A, géométrie brute** : `jobs.prepare_job` → `analyze/pipeline.prepare_part`
   (import `occ/loader`, analyse, bouchage de trous `occ/holes`, brep réécrit).
   Massive / sans peau → tétra direct. Le flag `profile` (profilé/lisse à section balayée)
   reste une information de triage : si les peaux réf/opp sont identifiées, tenter SC8R avant
   tout repli tétra.
2. `_sc8r_pass` : essais isolés (`strategy/runner.run_isolated`, multiprocessing *spawn*,
   timeout) de `strategy/attempt.run_attempt` ; recherche **gloutonne** : un seul levier par
   essai sur la meilleure recette (`alg` → `free` → `merge`), puis tailles et `subdiv`.
3. **Passe B** (workdir `microfix/`) si A échoue et si l'import a accepté la suppression des
   micro-arêtes (`Geometry.OCCFixSmallEdges`).
4. `_tet_fallback` : sources (brep préparé, STEP micro-fix 0,01 / 0,005, STEP brut) ×
   `TET_ALGOS`, chaque combinaison isolée (gmsh segfaulte parfois).
4bis. **Conseiller LLM : retiré** (sept. 2026, décision de Fred, contre-productif). Module
   `llm/`, `strategy/levers.py`, `[llm]`, options `--llm*` supprimés ; la montée gloutonne
   déterministe de `_sc8r_pass` reste seule. Sauvegarde : `../stepmesher_backup_20260928.zip`.
5. Export `io/inp_writer` (+ `write_inp_tet`), `io/exports` (.vtu, CSV), validation
   `io/inp_validator`, rapport JSON.

Modules clés :
- `analyze/skins.py` : rayons vectorisés (KDTree par classes de taille + Möller-Trumbore)
  → épaisseur et face opposée par face ; `exit_cos` filtre les surfaces de *sortie*
  (normale·d > cos). Classement peau/chant par face + propagation, 2-coloration.
  Une face n'est pas peau si son épaisseur < 1/3 de celle de sa face opposée (chant arrondi
  R ~ t/2 vu de biais), et un même côté ne se prolonge que par raccord tangent
  (< `patch_angle_deg`) : sinon un chant arrondi relie les deux peaux en une composante
  (part_029 : 0 face réf. / 43 opp. -> tétra).
- `analyze/classify.py` : constante / variable / massive, paliers par histogramme log pondéré,
  choix de la peau de référence (variable → la moins découpée).
- `mesh/quad.py` : plis transfinis (`_bend_sides`, `_ordered_loop` — **getCurveLoops /
  getBoundary ne sont pas ordonnés**), `h_gen` commun aux plis voisins, `apply_strategy`.
- `mesh/repair.py` : `flip_repair` (bascule + lissage, critère = pire quad remplacé sans
  dégrader les voisins), `fix_micro_edges` (glissement de nœuds).
- `mesh/offset.py` : normales CAD moyennées, onglet `1/cos(θ/2)`, rayons vers peau opposée +
   chants, avec second lancer vers la peau opposée si une tôle constante touche un chant avant
   75 % de son épaisseur ; `nearest_cad` (projection CAD exacte), nœuds sans cible = déplacement
   des voisins, `_untangle`.
- `mesh/quality.py` : critères durs / cibles, jacobien aux coins et déterminant trilineaire
   interne SC8R (grille 3x3x3), exemption « imposé par la CAD »
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
- Nettoyage OCP : un STEP sans **solide** (compound/shell) casse tout (analyse « massive »,
  épaisseur fausse, gmsh « aucun solide ») → `clean_step` exige `n_solids` inchangé.
  `ShapeFix_Wire.FixSmall` par contour retire bien les arêtes mais **détruit le solide** ;
  seul `ShapeFix_Wireframe.FixSmallEdges` (contexte global) le préserve.
- La stratégie `compound` globale plante / dépasse les délais sur CATIA : retirée des défauts.
- `threads > 1` : gmsh non déterministe (± quelques dizaines d'éléments).
- Les `.inp` sont écrits en ASCII pur (`to_ascii`) : pas d'accents dans Abaqus.
- `summary.csv` : `time_s` du repli tétra corrigé (le budget tétra ne masque plus T0).

## Points à confirmer au premier datacheck Abaqus

1. épaisseur de `*SHELL SECTION` (section unique, épaisseur constante = moyenne du maillage)
   vs épaisseur nodale des SC8R ;
2. `STACK DIRECTION=3` avec numérotation 1-4 / 5-8 ;
3. faces S1/S2 des surfaces ; ordre des nœuds milieux C3D10.

Orientation par élément (`*DISTRIBUTION` / `*ORIENTATION`) **retirée** du writer pour l'instant
(invalide telle quelle dans `*PART`, demande de Fred) ; les axes 1/2 par élément restent
écrits dans `_orientation.csv` pour la réintroduire plus tard.

## Pistes futures (par priorité proposée)

0. **PRIORITÉ ABSOLUE : qualité visuelle du maillage** (trop d'éléments dans les rayons,
   sauts de taille) — voir la section « PROCHAINE TÂCHE PRIORITAIRE » en tête de fichier.

1. **Relance et analyse de campagne complète** : vérifier d'abord l'intelligence effective du
   conseiller LLM sur les cas réels (priorité SC8R, utilisation des métriques dont jacobien
   interne, jamais de repli prématuré), puis collecter tous les `.json` / `.inp` et classer les
   causes (face non maillée, retournés aux raccords, reprojection, délais, tétras non massifs)
   avant de coder davantage.
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
9. ~~LLM local~~ : retiré (voir 4bis).
10. Surfaces de chants (`SURF_EDGES`) et sets par bord libre si utiles au chargement.
