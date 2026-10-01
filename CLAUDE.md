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
pytest -m "not real"                               # 141 tests synthétiques, ~1 min (20 cœurs)
pytest -m real                                     # 7 tests sur stps_test/, ~20 min
cp -r src /tmp/vN_src && tools/run_campaign.sh campagne/parts_stp_echelle result_vN 10 /tmp/vN_src
                                                   # campagne en parallèle (Linux), 1 processus/pièce
```

Environnements : conteneur 2 cœurs / 7 Go (anciens temps) ; poste Windows de Fred (AGENTS.md) ;
**machine Linux de Fred** (Fedora, 20 cœurs / 30 Go) : venv `.venv/` du projet (Python 3.12,
gmsh 4.15.2, cadquery-ocp) -> `.venv/bin/stepmesher`, le python système n'a aucun paquet.
Dossier du projet : `~/Téléchargements/stepmesher` (ex-`stepmesher-main`, renommé le 01/10/2026 :
**un venv ne survit pas au renommage de son dossier**, chemins absolus dans `.venv/bin/*` et le
`.pth` de l'installation éditable -> corrigés à la main ; lien `~/.local/bin/stepmesher` ->
`.venv/bin/stepmesher` pour la commande hors venv ; jamais de `pip install .` avec le python système).
**Campagnes de Fred** (`campagne/`) : `parts_stp_echelle` (44 pièces, ex-`stps_test` étendu) et
`steps_upper` (12 grandes pièces depuis le 01/10/2026 : cadres 2-7 m, 001 à 005 = cadres sœurs,
part_011 = coque supérieure 11 m / 128 m² ; `parts.csv` en liste 23, d'autres arriveront).
`stps_test/` n'existe pas sur cette machine (`pytest -m real` sans objet).
Fred visualise avec `mesh-viewer.html` (lit les `.inp`). **Ne jamais modifier `src/` pendant
une campagne** : les essais en sous-processus réimportent le code (lancer depuis une copie figée).

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

**29/09/2026 (result_v8, part_004)** : le « rayon à 14 éléments » de Fred n'était PAS un pli
(les 3 plis ont 3-5 rangées) mais un **coin arrondi du contour** (R 5 à 90°, vu dans le plan de
la tôle) : 12 segments de 0,74 mm, à cause de 2 facettes de chant BSpline de 1,03 x 2,17 mm à ses
bouts (arêtes de 1 mm sur la peau -> gradation arêtes courtes + parité 2 segments). Pour Fred,
« rayon » = pli OU coin du contour : même règle 3 él./90°.
- **Fait** : `quad.contour_arcs` (appelé dans `apply_strategy`, pas en subdiv) impose sur tout arc
  du contour de peau (courbe peau∩chant, rotation >= 15°, hors trous conservés) n = max(1 él./
  max_bend_angle_deg, L/h0), pair en full-quad -> **4 sur 90°** (3 impossible : gmsh full-quad
  exige un nombre pair par courbe). 004 : arc R5 12 -> 4 segments ; 122 tests OK.
- **Non résolu** : les 2 arêtes de 1 mm restent (2 segments de 0,5 mm aux bouts de l'arc).
  `BRepAlgoAPI_Defeaturing` d'une facette de chant quelconque (4 arêtes : 2 ~ t, 2 <= 1,5 mm)
  marche sur 4 facettes sur 6 de part_004 (une par une, validité vérifiée) mais PAS sur les
  2 du coin (solide invalide, même après ShapeFix_Shape ; ou aucun effet). `setCompound` 1D
  garde le nœud commun (vérifié). Nettoyage profond à 1,1 mm : accepté (valide, dV/V 8e-4)
  mais **replie des éléments dans le vide de l'encoche** sans que la qualité le voie -> NE PAS
  monter `occ_wireframe_precision_mm_deep`. Les contrôles d'arcs sur .vtu par
  `gmsh getClosestPoint` sur BSpline sont faux (renvoie l'extrémité).
- **CADUC (autre machine ; ni ce code ni `result_v8`/`result_v9` ne sont arrivés sur la machine
  Linux, voir « 29/09/2026 après-midi » ci-dessous)** — (29/09/2026 ~09:08) : campagne 44 pièces avec `contour_arcs` lancée ->
  `result_v9/` (journal `result_v9_run.log`, config par défaut). À faire ensuite :
  `python tools/compare_runs.py result_v8 result_v9`, vérifier statuts, nb d'éléments,
  small_pct/jump_pct, puis vues zoomées des coins arrondis (004 d'abord). Si la campagne a été
  coupée : la relancer (même commande, `-o result_v9`). Sauvegarde du code modifié :
  `~/stepmesher_backup_20260929_contour_arcs.zip`. En attente de Fred : (a) accepter ou non les
  2 éléments de 0,5 mm aux bouts de l'arc R5 de 004 ; (b) ajouter ou non la suppression OCP des
  facettes de chant quelconques (4/6 sur 004), à valider sur une campagne -> (b) : OUI (Fred,
  29/09 après-midi), fait.

**29/09/2026 après-midi (machine Linux, campagnes `campagne/`) — état du code réel** : le code
reçu était celui du 28/09 : `contour_arcs` décrit ci-dessus N'Y ÉTAIT PAS (réécrit), `llm/` et
`levers.py` encore présents (supprimés, accord de Fred). Référence refaite sur cette machine :
`result_base_echelle` (42/44, 012 et 021 FAILED_QUALITY, 29,8 % de petits éléments en moyenne) et
`result_base_upper` (000/009/022 en « plis libres » à 86-92 % de petits éléments, 001
FAILED_QUALITY, 011 FAILED). Causes trouvées et corrections (campagnes v1 à v5) :
- **arête CAD de ~1 mm sur un contour lisse -> 2 segments de 0,5 mm (parité full-quad) ->
  élancement 11-17 hors cibles -> recette nominale rejetée** au profit de « plis libres » ou
  x0,5 (3-4 fois plus d'éléments, part_016/017/012) : **`repair.relax_boundary`** (après
  `fix_micro_edges`) redistribue les nœuds le long du contour entre ancrages (coins > 25°,
  faces transfinies, trous, intérieur des arcs imposés), lissage local, accepté si l'élancement
  s'améliore sans dégrader le jacobien (`[mesh] boundary_relax*`) ; 016 : 1 017 -> 246 él. ;
- **nœuds lissés hors de la surface** (lissage laplacien 3D de `flip_repair` / relaxation : sur
  un pli R8, 4 nœuds à 1,7 mm DANS la matière -> écart d'épaisseur 0,53, part_021 ; upper 000) :
  `attempt.reproject_moved` ramène tout nœud déplacé sur sa face CAD (getClosestPoint) ;
- **plis coupés en tranches plus courtes que l'arc** (cadres upper : arc 9,8 mm, tranche 1,75 mm) :
  `_bend_sides(by_turn=True)` prend pour génératrices les courbes DROITES (rotation de tangente
  < 10°) ; avant, les arcs -> conflit de comptes, face libre, triangles + gauchissement ->
  « plis libres » gagnait (upper 022 : 31 747 -> 11 140 él.) ;
- **parité par courbe dans les chaînes découpées** : `_split_count(..., even=)` (plis, petites
  faces transfinies) : une micro-courbe partagée avec une face libre recevait 1 segment ->
  « 1D mesh cannot be divided by 2 » ; `structured_patches` : élancement <= 6 (sens long redécoupé) ;
- **`contour_arcs` réécrit** (après `merge_micro_curves`, arcs fusionnés ou < 0,5 h0 exclus) :
  PLAFOND « 3 éléments dans un rayon, pas plus » : n = max(L/h0, min(1 él./30°, L/(0,35 h0))),
  pair -> R4-R5 à 90° : 2 segments, R13 : 4 ; l'ancienne règle (4 segments forcés) mettait
  4 x 1,6 mm sur R4 (rosettes, 016). 004 : le coin R5 à 12 segments disparaît ;
- **facettes de chant** (Fred : « les supprimer ») : `clean._defeature_chant_fillets` généralisé
  (4 arêtes : 2 ~ t, 2 <= 3 mm, `occ_chant_facet_max_width_mm`) ; toutes d'un coup, sinon une
  par une (60 s). **Défaut de BRepAlgoAPI_Defeaturing** : il crée souvent des arêtes de 0-0,3 mm
  près des coins (014 : 10 créées) -> suppression refusée si elle crée une arête < t/4, si
  |dV| > 3 x aire x t des facettes ou si plus de faces disparaissent que demandé ; lancé en
  sous-processus avec délai `occ_chant_timeout_s` = 90 s (un seul appel : 334 s sur upper 018).
  Gain réel limité (1 à 5 facettes par pièce) : la relaxation du contour fait l'essentiel ;
- `occ_wireframe_precision_mm` 0,05 -> **0,1** : une courbe de 0,096 mm coupée en 2 (parité)
  donnait des arêtes de 0,048 < 0,05 (critère dur, 014, upper 001/018). Tétra : sources
  supplémentaires = STEP d'entrée nettoyé OCP 0,05 mm puis tel quel (`tet_sources(extra_steps)`),
  part_001 (0,027 < 0,03 au nettoyage 0,1 mm) ;
- décalage (`offset`) : rayon de bord libre qui touche un chant incliné avant 75 % de t et second
  rayon qui manque la peau opposée -> point le plus proche de la peau opposée si l'épaisseur est
  plausible (029 : 1,34 mm pour 2,71 sur un arc de contour) ; nœud relaxé dont le décalage dévie
  -> remis au sommet CAD ;
- **grandes pièces** : `flip_repair` borné à 120 s (sur 011 il tournait sans fin -> délais) ;
  budget par essai/pièce x (éléments estimés / `budget_ref_elements` = 10 000), plafonds 3 600 /
  14 400 s ; budget partagé entre variantes de géométrie (nettoyée 50 %, brute 75 %) ; pièce x5
  et plus : une variante qui passe suffit ; triangulation d'analyse plafonnée à
  `sample_size_max_mm` = 20 mm (1 % de 11 m = 110 mm -> erreur de reprojection 0,6) ; faces de
  plus de `large_face_elements` = 20 000 éléments en algorithme 2D 6 (l'algo 8 fait 26 000
  quads énormes et étirés sur une BSpline de 32 m² au lieu de 790 000) -> **upper 011 : 788 523
  SC8R, passe, ~16 min, 1,7 Go**.
Ancienne référence `result_v6_echelle` / `result_v6_upper` (011 copiée de v5, chemin de code
identique ; ancienne référence de cette machine : `result_base_*`) :
- echelle **43/44** (30 OK, 2 OK_APPROX, 11 OK_TET, 021 FAILED_QUALITY), petits éléments
  29,8 -> **15,8 %** en moyenne, SC8R total 62 070 -> 49 311 (016 1 017 -> 246, 017 1 355 -> 258,
  023 4 563 -> 1 508, 025 2 724 -> 1 399, 012 FAILED -> OK 803), ~11 min en parallèle (x11) ;
- upper **8/8** (base : 011 FAILED, 001 FAILED_QUALITY), petits 48,5 -> **16,8 %** ; 000 31 778 ->
  10 391, 009 33 331 -> 11 973, 022 31 747 -> 11 140, 001 -> OK_APPROX 20 572, **011 -> OK_APPROX
  789 343 SC8R (1 essai de 855 s)** ; ~13 min (x7) + 18 min pour 011 ;
- rendus avant/après : `result_v6_*/avant_apres/`.
**Reste** : 021 (ligne CAD diagonale entre faces tangentes -> aiguilles, 7 hors
cibles pour 5 ; la fusion en composite fait pire) ; temps x1,8 sur les petites pièces (suppression
des facettes une par une + variantes) ; petits éléments restants = gradation autour des arêtes
courtes intérieures (fins de lignes de tangence) et coins en chanfrein réels.

**29/09/2026 soir — retour de Fred sur upper part_001 (pattes latérales 7,4 m x 36 mm « pas du
tout régulières ») ; ordre validé : 1) détecteur, 2) correction des lanières** :
- **Détecteur (fait, SIGNALEMENT seul, pas d'ELSET — décision de Fred)** : `quality.topology`
  compte les nœuds intérieurs à 3 ou 5+ quads (« étoiles ») dans les faces qui DEVRAIENT être en
  rangées régulières = `quad.regular_expected_faces` (une boucle, 4 coins réels = virage >= 30°
  entre courbes >= 0,5 h0, côtés opposés rapport <= 2 -> pattes, lanières, plis, rectangles ; pas
  les âmes de forme libre, où quelques étoiles sont normales). Sorties : `metrics.topology`
  (faces fautives + position), log WARNING « maillage IRRÉGULIER », `summary.csv`
  (`irregular_faces`, `irregular_nodes`), `compare_runs.py` (« REGRESSION topologie », colonne i),
  pénalité de départage += `regularity_topology_weight` x %. Les critères par élément (petits,
  sauts) ne voyaient rien : 001 = 400 étoiles pour 10 % de petits / 1,2 % de sauts. Sur v7 : 001
  (400), echelle 010 (78), upper 000 (64), echelle 018 (51, face 214 x 140), 020/021 (47)... ;
  petits clips : 0. Tests `tests/test_topology.py`.
- **Cause sur 001** : la patte a 12 courbes (un grand côté = pli en 7 faces, l'autre = 1 courbe,
  bouts en biais 32°/148° + chanfrein 1,9 mm) -> lanière rejetée (2 plus longues courbes prises
  pour grands côtés) ET comptes des 2 plis voisins incompatibles.
- **Planification harmonisée (faite, `[mesh] structured_harmonize = true`)** :
  `quad.structured_plan` : plis + lanières + faces à 4 côtés (jusqu'à
  `structured_patch_max_side_frac` = 8 h0) inventoriés ENSEMBLE, côtés par les 4 coins réels
  (`_rect_sides` ; coin = sommet de plus fort virage, jamais un raccord tangent), repli sur
  l'ancienne détection (`_strip_plan_legacy`, `_patch_plan_legacy` : lanière à bouts arrondis
  tangents, 015 face 12) ; comptes harmonisés par AJOUT seulement (côtés opposés égaux, parité
  des courbes bordant une face libre) ; face insoluble -> libre ; pas de solution -> ancien
  enchaînement. `flip_repair` : pas de bascule dans une face transfinie sauf quad < 0,1
  (sinon étoiles). 001 (géométrie profonde) : 29 -> 33 faces structurées, 0 hors cibles.
- **Reste sur 001 : les 2 pattes** — leur bout se prolonge par une FACETTE DE CHANT de 1,9 mm
  tangente au bout ; coin transfini au raccord tangent -> quad plat retourné ; coin au vrai
  virage -> rangée de 1,9 mm sur 7,4 m ; suppression OCC de la facette : 178 s et 0/10 facettes ;
  courbe composite gmsh : garde le nœud ; `addTrimmedSurface` (reconstruire la face avec bout +
  chanfrein fusionnés) : « Could not create wire ». Pattes laissées libres (signalées). Piste :
  topologie virtuelle sur la peau (OCP : coque de peau, fusion des arêtes tangentes du contour)
  ou affaissement de la rangée fine après maillage — À DISCUTER avec Fred.
- **Topologie virtuelle (faite, choix de Fred « la solution propre », `[mesh] virtual_topology`)** :
  `occ/virtual.py` : sur les seules faces de peau (modèle de MAILLAGE, solide intact), courbes du
  contour extérieur reliées par un sommet parasite (sur une seule face de peau, virage < 15°, une
  des deux courbes < 3 mm) concaténées en BSpline EXACTE (GeomConvert_CompCurveToBSplineCurve),
  face reconstruite sur la même surface (sens du contour choisi pour une aire > 0), autres arêtes
  réutilisées (partage intact). Construit en fin d'analyse (`pipeline._virtual_skin`, ~1 s) :
  `pa.skin_brep`, `skin_face_map` (face du modèle de peau -> face de brep), `skin_curve_map`
  (courbe de brep -> courbe du modèle de peau, par distance point-segment 0,05 mm ; rejet si une
  courbe de peau n'a pas de correspondance). L'essai maille dans cette numérotation
  (`quad.skin_view` : faces hors peau en numéros NÉGATIFS ; `attempt._recipe_in`) puis revient à
  celle de pa.brep juste avant le décalage (quad_face, listes structurées, faces attendues
  régulières, faces non maillées). Tests `tests/test_virtual.py`.
  Effet (v11 vs v10) : **echelle 44/44 (021 passe)**, petits 15,7 -> 9,2 %, SC8R 49 167 -> 45 502,
  temps cumulé 78 -> 43 min, 009 332 -> 169 él., 031/033 passent à x1 au lieu de x0,7 (-40 %) ;
  upper : **001 0 étoile (pattes régulières, 3 rangées)**, 018 72 -> 45 ; 020/021 40 -> 85 = face 7
  (âme trapézoïdale 2,2 m x 161-258 mm, 4 coins, trop large pour une lanière, maillée en libre :
  tirage différent), signalée dans les deux versions.
- **Redressement des colonnes (fait, accord de Fred, `[mesh] align_columns`)** : le transfini
  relie le k-ième nœud d'un côté au k-ième de l'autre -> côtés de longueurs différentes (pattes de
  001 : 7 405 / 7 330 mm, bouts en biais) = colonnes penchées ~45° sur 7 m (décalage 20-60 mm).
  `repair.align_columns` (après fix_micro_edges, AVANT flip_repair) : grille reconstruite
  (`structured_grid`), nœuds d'un grand côté glissés sur la polyligne du côté pour faire face à
  ceux de l'autre, transition près des bouts (2 x largeur + 3 x décalage), nœuds sur un sommet
  CAD fixes, propagation de bande en bande par les côtés partagés (patte -> pli -> âme ...),
  intérieurs par interpolation des déplacements, reprojection ensuite ; annulé si la qualité
  baisse. 001 : décalage 0,0 mm de la colonne 20 à 759, jacobien p05 0,54 -> 0,99 (min 0,377
  -> 0,351, angle min 20,8°). v12 vs v11 : 0 régression echelle, 018 étoiles 45 -> 18.
  **Décision de Fred** : grandes faces à 4 coins trop larges pour une lanière (020/021 face 7,
  âme 2,2 m x 161-258 mm) laissées en maillage libre pour l'instant (signalées).
- Ancienne référence `result_v12_echelle` / `result_v12_upper` (+ redressement). Rendus :
  `result_v12_upper/avant_apres/` (patte 86 en 3 étapes v6 / v11 / v12). Avant :
  `result_v11_*` (supprimé) = topologie virtuelle ; `result_v10_*` (supprimé) = détecteur + planification harmonisée,
  0 régression vs v7 (`compare_runs.py`) : echelle 43/44 (021), petits 15,7 %, 010 étoiles 78 -> 62 ;
  upper 8/8, petits 16,8 -> 10,4 %, étoiles 000 64 -> 0, 009 20 -> 0, 022 7 -> 0, 001 400 -> 104
  (les 2 pattes à chanfrein), 018 185 -> 72 ; +12-15 % d'éléments sur 001/018 (les lanières
  prennent le pas des plis le long du profilé).

**29/09/2026 soir (retour de Fred sur v12) — 3 défauts** :
- **Maillage DISCONTINU sur une traverse (echelle 021, intercostale)** — le plus grave : fente de
  16 mm entre pli et semelle, 31 nœuds de bord superposés (paires à 0,004 mm), un seul morceau
  (`mesh_pieces` aveugle), .inp validé OK. Cause : la **passe B** (micro-arêtes supprimées à
  l'import gmsh, `Geometry.OCCFixSmallEdges`) découd deux faces de peau ; ses essais passaient et
  gagnaient. **Détecteur (fait, critère DUR)** : `quality.mesh_cracks` (nœud de bord à moins de
  `crack_tol_mm` = 0,05 d'une autre arête libre non atteignable en suivant le bord sur moins de
  `crack_min_path_mm` = 1 ; + arêtes partagées par 3+ éléments), dans `evaluate` (raison
  « maillage DISCONTINU », score -1000) et dans `inp_validator` (idem + paires de nœuds confondus
  < 0,005 mm sans élément commun, tous types d'éléments). v12 : seule 021 touchée (0 faux positif
  sur 44 + 8 pièces, v6 propre). Tests `tests/test_cracks.py`.
- **Rayon trop découpé (echelle 013, fond d'encoche R5 à 12 segments de 0,67 mm)** : les 2 bouts
  de 0,86 mm (restes de facettes de chant) sont TANGENTS à l'arc, mais la topologie virtuelle
  refusait la face (« concaténation impossible ») : tolérance fixe 1e-6 pour
  `GeomConvert_CompCurveToBSplineCurve.Add`, écarts CATIA de 5e-6 à 1e-5 mm. **Corrigé** :
  tolérance = max(`JOIN_TOL_MIN_MM` = 1e-4, 2 x tolérance du sommet) (`occ/virtual.py`). 013 :
  arc à 4 éléments, petits 27 -> 13 %, jacobien min 0,17 -> 0,37 ; 014 petits 23 -> 4 %, 000
  26 -> 4 %. 021 : la géométrie brute passe désormais sans passe B (1 497 él., continu).
- **Petites arêtes sur upper 022 = SOYAGES** (confirmé par Fred) : décalage de 0,9 mm de la
  semelle sur 7,2 mm (S en 3 tranches : arc 10°, droite, arc -10°), au droit des encoches ; le pli
  âme/semelle ET la semelle sont découpés en tranches BSpline de 0,9 à 3,4 mm ; chaque courbe CAO
  porte une colonne de nœuds, parité full-quad -> 6 colonnes de 0,5-1,7 mm dans la rampe (Fred :
  « trop, si possible réduire »). Explique ~80 % des quads minuscules des cadres upper 000, 009,
  022 ; invisible des critères de régularité (plis transfinis exemptés). **gmsh 4.15
  `setCompound(2, ...)` garde les nœuds des courbes internes** (vérifié, comme en 1D).
  **Fait : fusion des tranches (`occ/virtual.merge_slices`, `[mesh] virtual_slices*`)**, dans le
  modèle de peau seulement : tranche = face à 4 arêtes dont 2 côtés opposés < 0,4 h0 ; tranches
  voisines par leurs côtés longs et tangentes (< 5°) -> UNE face, surface de Coons
  (`GeomFill_BSplineCurves`, courbes montées au degré 3) sur les 4 coins réels (plus forts
  virages ; un côté peut garder un sommet), bords courts consécutifs concaténés en BSpline exacte,
  la MÊME arête dans les faces voisines (âme reconstruite sur son plan) ; refus si écart > 0,05 mm
  dans les deux sens (points des tranches -> surface ET points de la face -> tranches : la seule
  aire ne suffit pas) ; un groupe refusé = recalcul sans lui (sinon arête concaténée d'un seul
  côté = fissure). APRÈS la simplification des contours (un sommet parasite sur le bord libre
  d'une tranche lui donnait 5 arêtes). `skin_face_groups` {représentant: tranches} ; `skin_view` :
  o2n envoie chaque tranche sur la face fusionnée ; `attempt.reclassify_groups` rend chaque quad à
  sa tranche d'origine (triangulation d'analyse) avant le décalage ; listes structurées étendues
  aux tranches. 022 : 81 -> 49 faces de peau, bord libre identique (partage intact), écart
  <= 0,022 mm ; 2 colonnes dans la rampe au lieu de 6. Tests `tests/test_virtual.py`.
- Campagnes : `result_v13_*` (détecteur de fissure + tolérance de raccord ; 0 régression vs v12,
  echelle 000/013/014 plus réguliers, 021 continu ; upper identique) ; rendus
  `result_v13_echelle/avant_apres/`.
- Ancienne référence `result_v14_echelle` / `result_v14_upper` (+ fusion des tranches) : echelle
  identique à v13 (aucune tranche fusionnée) ; upper 000 10 656 -> 5 026 SC8R (petits 33 -> 7 %,
  36 faces fusionnées), 009 10 578 -> 9 361 (29 -> 17 %), 022 10 572 -> 6 704 (15 -> 3 %) ;
  hors cibles 009 79 -> 41, 022 43 -> 6 ; jacobien min un peu plus bas (000 0,28 -> 0,23, 009
  0,23 -> 0,18, 022 0,38 -> 0,32 : maillage libre ailleurs, PAS dans les rampes fusionnées),
  signalé « REGRESSION jacobien » par compare_runs ; 52 .inp continus et valides. Rendus
  `result_v14_upper/avant_apres/`.
- **Fred (29/09 soir)** : corrections 021 (fissure) et 013 (rayon) jugées correctes ; soyages :
  « ça ira pour le moment ». Vocabulaire : « traverses » = famille `intercostale` (echelle
  021-033) ; « rayon » = pli OU coin arrondi du contour (fond d'encoche compris) ; « edges trop
  petits » = colonnes minuscules imposées par des courbes CAO rapprochées ; « maillage pas
  continu » = faces qui ne partagent pas leurs nœuds (défaut le plus grave pour lui).
- **Reste (non traité)** : upper 009 garde 17 % de petits éléments, AUTRE cause que les soyages
  (non analysée) ; echelle 023/029 : quelques tranches de pli isolées (pas de groupe de 2+, non
  fusionnées) ; jacobien min un peu plus bas sur 000/009/022 (voir ci-dessus) ; la passe B
  (`OCCFixSmallEdges` gmsh) peut toujours découdre des faces : désormais rejetée par le critère de
  fissure, pas empêchée à la source.
- Outils de vérification utilisés : `stepmesher validate` (continuité incluse) sur tous les .inp
  d'une campagne ; `tools/compare_runs.py vN vN+1` ; rendu avant/après d'une zone :
  `tools/render_cmp.py avant.inp apres.inp out.png cx,cy,cz rayon [vue|-] [titre_avant titre_apres]`
  (remis dans `tools/` le 01/10 avec `readinp.py` ; `result_pbm/` n'existe plus).

**01/10/2026 — « ça ne marche plus sur d'autres pièces très similaires » (Fred : « le script doit
être robuste »)** : `steps_upper` passe de 8 à 12 pièces (002 à 005, sœurs de 001). Avec le code
v14 : 002, 003, 004 FAILED_QUALITY (19 min chacune, budget épuisé), 005 OK (`result_v14_new4`).
Trois causes, aucune propre à ces pièces :
- **Aire d'une face longue et étroite fausse de 0,2 à 1,2 %** (intégration PAR DÉFAUT de
  `BRepGProp.SurfaceProperties` : âme de 6,9 m x 24 mm = 167 657 ou 165 223 mm² selon la pièce,
  pour 167 279 ; l'erreur change avec le découpage du contour). `occ/virtual.build_virtual_skin`
  comparait l'aire de la face reconstruite à l'ancienne à 1e-3 près -> « face invalide », âme
  laissée en maillage LIBRE (éléments de 37-50 mm en biais, retournés ; 003 et 004). **Corrigé** :
  intégration adaptative (`AREA_EPS = 1e-6`, `area_exact`, aussi dans `merge_slices`) ; l'aire par
  défaut ne sert plus qu'à retrouver les faces de gmsh (`getMass` intègre de la même façon).
  Effet de bord utile sur echelle (faces jusque-là refusées à tort) : 004, 005, 011, 018.
- **`skin_maps` : polyligne à pas uniforme en paramètre (401 points)** : dans une longue courbe
  concaténée (485 mm, echelle 018), un petit arc n'avait que quelques points -> flèche > 0,05 mm ->
  « courbes de peau sans correspondance » -> TOUTE la topologie virtuelle de la pièce rejetée
  (démasqué par la correction d'aire). **Corrigé** : raffinement là où la corde s'écarte de la
  courbe. 018 : 827 -> 640 él., petits 10,9 -> 0,7 %, étoiles 54 -> 0 (face 214 x 140).
- **Faces-lanières de 0,1 mm de large dans la peau** (002 : 24, 004 : 12 ; 7 à 25 mm de long, en
  travers des semelles et des plis au droit des marches d'épaisseur, x = 6238 / 6459 / 9315 /
  9370 / 12472 / 12545 ; certaines triangulaires, 0 -> 0,1 mm). Maillées : 2 segments de 0,05 mm
  dans la largeur (parité) -> critère dur d'arête mini + hexa écrasés. Nettoyage OCP 0,1 mm : leurs
  bouts (0,0999 mm) sont effondrés -> faces d'AIRE NULLE à 2 courbes, « non triangulées,
  ignorées » à l'analyse puis « face de référence non maillée » à tous les essais. Couture OCC
  (`BRepBuilderAPI_Sewing` après retrait des lanières) : 36 bords libres, solide invalide, dV/V
  0,5-1 % -> abandonnée. **Fait : couture AU MAILLAGE (`[mesh] sliver_seam_max_width_mm` = 0,3)** :
  - `analyze/features.detect_sliver_seams` (dans `prepare_part`, sur toutes les faces du brep, y
    compris écrasées / non triangulées) : face à une boucle, largeur moyenne 2A/P < seuil, contour
    = DEUX chaînes de courbes longues (>= 5 x seuil) séparées par des bouts courts ou un demi-tour
    (> 120° : pointe d'un triangle, face à 2 courbes), longueurs voisines, bouts rapprochés, les
    deux chaînes bordant la peau de référence (lanières accolées : chaînes extrêmes). La lanière
    est RETIRÉE de `ref_faces` / `flank_faces` / `bends` (donc du modèle de peau) ; `pa.seams`
    (face, a, b, width, length), `pa.sliver_faces`, message dans le log ;
  - `quad.skin_view` traduit les courbes (modèle de peau) ; `quad.seam_chains` ordonne et oriente
    les deux chaînes et repère les bouts de lanière restés dans le modèle (`links`) ;
  - `quad.structured_plan(seams=)` : la couture est une pseudo-face (numéro < 0, kind « couture »)
    à deux côtés opposés = mêmes totaux, nombres PAIRS, sans surface ; hors planification (plis
    libres, pas de solution) : `quad.seam_counts`, avant les faces structurées ;
  - `quad.weld_seams` (juste après `extract_reference_mesh`, avant toute retouche) : k-ième nœud
    de b remplacé par le k-ième de a (qui reste sur sa courbe ; déplacement <= largeur, 0,18 mm
    max ici) ; refus si comptes différents ou glissement > 0,45 x pas local. **Bout de lanière
    posé sur le bord d'une face NON coupée** (l'âme) : son quad se réduit à un triangle ->
    `_collapse_chords` : deux triangles accolés autour du nœud soudé (face libre, 2 segments sur
    le bout) réunis en un quad ; sinon la colonne est refermée de proche en proche jusqu'à un
    bord (« chord collapse », nœud au milieu sauf nœud épinglé = sommet CAO / couture). Jamais de
    quad retiré sans refermer : ça laissait un TROU triangulaire de 3 x 8 mm que ni
    `mesh_cracks` ni le validateur ne voient ;
  - `attempt` : une couture non soudée = échec explicite « maillage DISCONTINU : n couture(s) de
    face-lanière non soudée(s) » (la fente de 0,1 mm est plus large que `crack_tol_mm` = 0,05).
  Tests `tests/test_seams.py` (détection, soudure en rangées et en libre, colonne refermée).
- **RÉFÉRENCE = `result_v15_echelle` / `result_v15_upper`** (01/10/2026) :
  - upper **12/12** (1 OK, 11 OK_APPROX), 0 régression vs v14 sur les 8 anciennes ; 002, 003, 004,
    005 : **22 740 SC8R chacune, 1 essai (`conform×1`, géométrie nettoyée), jacobien min 0,34-0,36,
    0 % de petits éléments, 0 étoile**, ~3,5 à 4 min ; 018 : 21 014 -> 19 544, étoiles 18 -> 0 ;
    011 identique (789 343, 867 s) ; 12 .inp valides et continus ;
  - echelle **44/44** identique en statuts, SC8R 45 409 -> 44 826, petits 7,5 -> 6,1 %, étoiles
    120 -> 66 ; changent seulement 004 (981 -> 900, petits 17,5 -> 7,6 %), 005 (350 -> 320,
    12,3 -> 1,8 %), 011 (631 -> 564, 20,6 -> 6,6 %), 018 ;
  - rendus `result_v15_upper/avant_apres/` (002, 003, 004) et `result_v15_echelle/avant_apres/018.png`.
- **Reste / à savoir** : (a) le seuil 0,3 mm est absolu, non testé au-delà de 0,18 mm de large ;
  une lanière en BORD de peau (une chaîne sur un chant) ou sans deux chaînes nettes n'est pas
  traitée (reste une face ordinaire) ; (b) le repli tétra ne profite pas des coutures (lanières
  toujours dans le solide : « aucun maillage tétraédrique » sur 002/004 avec l'ancien code) ;
  (c) une géométrie nettoyée inutilisable consomme toujours 50 % du budget avant la brute ;
  (d) quand la topologie virtuelle refuse UNE face (`skin_info.messages`), une lanière de 7 m
  peut rester libre sans autre alerte que les étoiles : à surveiller sur les prochaines pièces ;
  (e) echelle 018 garde un nœud de maillage libre au milieu de la grande face (fin d'une ligne
  CAO intérieure), non signalé ; (f) upper 020/021 face 7 (85 étoiles) et 009 (17 % de petits)
  inchangés.

Piste de travail du 28/09 (points 1 à 3 faits depuis : diagnostic, régularité, topologie) :
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
   `llm/`, `strategy/levers.py`, `tests/test_llm.py`, `[llm]`, options `--llm*` supprimés
   (les fichiers traînaient encore dans le code du 28/09 : effacés le 29/09) ; la montée
   gloutonne déterministe de `_sc8r_pass` reste seule.
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
