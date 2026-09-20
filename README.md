# stepmesher — mailleur local STEP → Abaqus

Outil Python 100 % local (gmsh + numpy + scipy, aucune API cloud) qui maille des pièces STEP
individuelles pour Abaqus, avec contrôle qualité automatique et export `.inp`. Le choix de
l'élément dépend de la pièce : **SC8R, un élément dans l'épaisseur**, pour les pièces minces
(tôles pliées, cas majoritaire), tétraèdres pour le reste. Les pièces qui ne relèvent pas du
modèle « deux peaux » (pièces massives, nervures, jonctions en T) basculent sur un
**repli tétraédrique C3D10** (C3D4 en option).

> **État** : les 5 pièces réelles de `stps_test/` sortent avec un maillage valide
> (voir § 9). Tôle constante : `OK`. Épaisseur variable : `OK_APPROX`. Repli tétra : `OK_TET`.
> Pas encore faits : mémoire de recettes SQLite, LLM optionnel, zones d'épaisseur alignées
> sur le maillage (voir § 10).

---

## 1. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # gmsh, numpy, scipy
pip install pytest          # pour les tests
```

Sous Linux, le module `gmsh` de pip a besoin de quelques bibliothèques système
(sinon `OSError: libGLU.so.1` / `libXft.so.2` à l'import) :

```bash
sudo apt-get install libglu1-mesa libxft2 libxinerama1 libxcursor1 libfontconfig1 libxrender1
```

## 2. Utilisation

```bash
# mailler un fichier, plusieurs fichiers ou un dossier (récursif avec -r)
stepmesher mesh piece.stp -o out/
stepmesher mesh stps_test/ -o out/ -r

# analyse seule : épaisseurs, paliers, peaux, plis, trous, temps
stepmesher inspect piece.stp

# contrôler un .inp
stepmesher validate out/piece.inp

# pièces synthétiques de test
stepmesher gen-testdata synth/

# configuration par défaut (à copier puis passer avec --config)
stepmesher dump-config > ma_config.toml
```

Options utiles de `mesh` :

| Option | Effet |
|---|---|
| `--config fichier.toml` | surcharge des paramètres (seules les clés présentes changent) |
| `--reference-skin inner\|outer` | peau maillée (défaut : intérieure) |
| `--no-fill-holes` / `--fill-holes MM` | désactive le bouchage / fixe le diamètre seuil |
| `--allow-wedge PCT` | tolère PCT % de triangles, exportés en SC6R |
| `--strategies conform,subdiv` | ordre des stratégies de maillage quad |
| `--tet-order 1\|2` | repli tétra en C3D4 ou C3D10 (défaut 2) |
| `--no-tet` | pas de repli tétra (échec SC8R = `FAILED_QUALITY`) |
| `--llm` / `--llm-url URL` | conseiller LLM local (§ 7) |
| `--threads N` | threads gmsh |
| `--keep-work` | conserve les essais intermédiaires (`out/.work/`) |

Code retour : 0 si toutes les pièces sont `OK`, `OK_APPROX` ou `OK_TET`, 1 sinon.

## 3. Sorties (par pièce, dans `-o`)

| Fichier | Contenu |
|---|---|
| `<pièce>.inp` | maillage Abaqus (voir § 6) |
| `<pièce>.vtu` | même maillage pour ParaView : épaisseur, jacobien, zone, face CAD, normales, axe 1, éléments hors cible |
| `<pièce>_orientation.csv` | SC8R : par élément, normale, direction d'empilement, axe local 1, épaisseur, zone |
| `<pièce>.json` | rapport complet : analyse, nettoyage, recette, métriques, **tous les essais et la raison de chaque échec**, durées |
| `<pièce>.log` | journal lisible |
| `summary.csv` | synthèse du lot, une ligne par pièce (statut, type d'élément, nb d'éléments, qualité, temps) |

Si aucun essai ne passe les critères, le meilleur essai est tout de même exporté sous
`<pièce>.FAILED.inp` (statut `FAILED_QUALITY`) pour diagnostic.

Statuts : `OK` (SC8R, tôle constante), `OK_APPROX` (SC8R, épaisseur variable, transitions d'épaisseur lissées),
`OK_TET` (repli C3D10/C3D4), `FAILED_QUALITY`, `FAILED`, `ANALYSIS_FAILED`, `INVALID_INP`.

## 4. Principe

### 4.1 Nettoyage du STEP et tri des surfaces

1. **Import non destructif.** Correction des faces dégénérées. Tout nettoyage est rejeté
   (géométrie brute conservée) si le nombre de solides change ou si le volume varie de plus
   de `healing.max_volume_change` (1e-3 relatif).
2. **Micro-arêtes** (bouts d'arêtes CATIA de quelques centièmes de mm). Deux traitements :
   - *passe A (géométrie brute)* : les micro-arêtes restent dans la CAD, mais les nœuds du
     maillage qui s'y trouvent sont **glissés** le long de l'arête voisine colinéaire, et un
     élément dont le bord est plus court que `micro_edge_ratio × min_size_mm` est considéré
     comme **imposé par la CAD** (exclu des critères cibles, jamais des critères durs) ;
   - *passe B (géométrie corrigée)*, seulement si la passe A échoue : les micro-arêtes sont
     supprimées à l'import (`Geometry.OCCFixSmallEdges`, tolérances `healing.small_edge_tol_mm`),
     avec le même contrôle de volume.
   Supprimer systématiquement les micro-arêtes à l'import casse certaines faces BSpline
   (face non maillable, surfaces auto-intersectantes pour le tétra) : c'est pourquoi la
   géométrie brute est essayée d'abord.
3. **Tri des surfaces utiles.** Rayons lancés vers l'intérieur de la matière depuis une
   triangulation grossière (KDTree par classes de taille + Möller-Trumbore vectorisé, aucune
   requête OCC point par point) : épaisseur locale et face opposée de chaque face. Chaque face
   est classée **peau** (a une face opposée à distance plausible) ou **chant** (flanc
   d'épaisseur, bord biseauté), puis propagation aux faces tangentes d'épaisseur cohérente
   (plis, lanières CATIA) et répartition des peaux en deux côtés par 2-coloration.
   Seule la peau de référence est maillée ; l'autre peau sert de cible au décalage ; les chants
   ne sont jamais maillés.
4. **Plaque ou pas plaque.** Classement *tôle constante* / *épaisseur variable* (paliers
   détectés par histogramme pondéré) / *massive* (pas de peaux appariées, ou volume
   incompatible avec aire × épaisseur). Une pièce massive part directement en tétra.
5. **Trous** : les trous débouchants sous le seuil (`holes.max_diameter_mm`,
   `max_diameter_frac` × diagonale) sont bouchés et listés dans le JSON (centre, axe,
   diamètre, méthode). Voie rapide : reconstruction des faces sans leurs boucles puis
   recouture ; repli : *defeaturing* OCC.

### 4.2 Maillage SC8R

6. **Peau de référence** : l'intérieure (aire minimale) pour une tôle, afin que le décalage
   diverge aux plis ; la moins fragmentée (non usinée) pour une pièce à épaisseur variable.
7. **Maillage quad** de la peau de référence (Frontal-Delaunay quads + blossom full-quad),
   taille `clamp(min(size_frac × diagonale, k × épaisseur), min, max)`, raffinée aux plis,
   trous conservés et arêtes courtes. Plis à 4 côtés (ou à côtés chaînés) en **maillage
   transfini** : même nombre d'éléments sur les génératrices de plis voisins, pour que les
   bandes entre plis restent régulières.
8. **Réparation locale** : bascule d'arête + lissage autour des quads de qualité < 0,2,
   acceptée seulement si le pire quad s'améliore sans dégrader ses voisins.
9. **Décalage** de chaque nœud selon la normale CAD exacte (bissectrice et onglet
   `1/cos(θ/2)` aux angles vifs), rayon jusqu'à la peau opposée ou aux chants de sortie, puis
   **projection exacte** sur la surface CAD. Nœuds sans cible : déplacement moyen de leurs
   voisins, puis projection. Un démêlage local (lissage des déplacements) corrige les éléments
   retournés aux raccords.
10. **Contrôle qualité**, puis export.

### 4.3 Recettes adaptatives

Le premier essai utilise la recette par défaut. Si un essai échoue, les faces CAD qui portent
les éléments fautifs sont identifiées et **un seul levier** est appliqué à la meilleure
recette obtenue jusque-là (recherche gloutonne), dans cet ordre :

| Levier | Effet |
|---|---|
| `alg` | change l'algorithme 2D des faces fautives (6 → 1 → 5) |
| `free` | libère les plis transfinis impliqués (maillage libre) |
| `merge` | fusionne localement les faces fautives en surface composite |

Puis les variantes de taille (`size_variants`) et la stratégie `subdiv` (100 % quads garanti,
qualité moindre). Libellés dans le rapport : `conform+5 faces remaillées×1`,
`conform-libre×0.7`, etc.

Ordre complet pour une pièce : passe A (brute) → passe B (micro-arêtes corrigées, si la
passe A échoue et que la correction a été acceptée à l'import) → repli tétra.
Chaque essai tourne dans un **sous-processus isolé avec délai** : un plantage ou un blocage
de gmsh devient un simple essai échoué.

### 4.4 Repli tétraédrique

Déclenché pour une pièce massive, ou quand toutes les recettes SC8R ont échoué. Sources
géométriques essayées dans l'ordre : géométrie préparée, STEP avec correction de micro-arêtes
0,01 puis 0,005 mm, STEP brut ; trois combinaisons d'algorithmes gmsh par source. Chaque
combinaison est isolée (gmsh peut planter sur une surface auto-intersectante).

- C3D10 : nœuds milieux sur la CAD si tous les éléments courbes restent valides, sinon sur
  les arêtes droites (signalé dans le JSON) ; numérotation Abaqus vérifiée par géométrie.
- Critères : aucun volume négatif, qualité de forme (rayon inscrit normalisé) ≥
  `tet.min_shape_quality` hors éléments imposés par une micro-géométrie CAD (arête < 10 % de
  la taille mini), écart de volume maillage / CAD reporté.

## 5. Critères de qualité SC8R (`[quality]`)

- **Critères durs**, vérifiés sur 100 % des éléments : aucun élément retourné, jacobien
  normalisé ≥ `hard_min_scaled_jacobian` (0,1), 100 % quads (sauf `allow_wedge_pct`), erreur de
  reprojection ≤ 5 % de l'épaisseur, et pour une tôle constante écart d'épaisseur ≤ 15 %.
- **Critères cibles** : jacobien ≥ 0,2, élancement ≤ 10, angles entre 20° et 160°,
  gauchissement ≤ 20°. Au plus `soft_violation_pct` (**0,2 %**) des éléments peuvent être hors
  cible ; ils sont regroupés dans `ES_QUALITY_WARN` et listés dans le JSON.
  `soft_violation_pct = 0` rend tous les critères stricts.

Pourquoi deux niveaux : sur des pièces CATIA réelles, quelques quads à angle presque plat se
forment au contact des arcs de bord courts (dégagements de pli) ; l'angle y est imposé par la
géométrie.

## 6. Contenu du `.inp`

SC8R :
- `*NODE`, `*ELEMENT, TYPE=SC8R` (et `SC6R` si `--allow-wedge`) ;
- node sets : `NS_SKIN_REF`, `NS_SKIN_OPP`, `NS_FREE_EDGES`, `NS_HOLE_EDGES` ;
- element sets : `ES_ALL`, `ES_BENDS`, `ES_FLAT`, `ES_ZONE_<i>` (une zone par palier
  d'épaisseur), `ES_QUALITY_WARN` ;
- surfaces des **deux peaux externes** (toutes deux complètes, chants exclus) :
  `SURF_INNER` / `SURF_OUTER` (peau intérieure = côté concave des plis, d'aire la plus
  faible ; si l'écart d'aire est < 0,5 %, pièce quasi plane : le `.inp` et le JSON
  signalent que la distinction intérieure/extérieure est peu significative), et les mêmes peaux sous
  les noms `SURF_REF` (faces S1, nœuds 1-4) / `SURF_OPP` (faces S2, nœuds 5-8). Le JSON
  indique quelle face (S1/S2) porte chaque surface et les aires ;
- orientation par élément : `*DISTRIBUTION` (axes 1 et 2) + `*ORIENTATION` ;
- une `*SHELL SECTION` par zone : `MATERIAL=TBD`, `ORIENTATION=ORI_ELEM`, `STACK DIRECTION=3`.

Tétra : `*ELEMENT, TYPE=C3D10` (ou C3D4), `NS_SKIN`, `ES_ALL`, `ES_QUALITY_WARN`,
`*SOLID SECTION, MATERIAL=TBD`.

**Aucun bloc matériau** : définir `*MATERIAL, NAME=TBD` (ou renommer) avant calcul.
Fichiers en ASCII pur (pas d'accents) pour Abaqus.

## 7. Conseiller LLM local (optionnel)

Quand une recette échoue, le prochain essai est choisi soit par les règles déterministes
(§ 4.3), soit par un **LLM local** servi par llama-server. Désactivé par défaut.

```bash
llama-server -m qwen3-4b-instruct-q4_k_m.gguf -c 8192 -ngl 99 -fa --port 8080
stepmesher mesh piece.stp -o out/ --llm            # ou --llm-url http://127.0.0.1:8080
```

- **Entrée** : résumé de la pièce (nature, paliers, nombre de faces, plis, trous) et
  historique des essais (recette, raison de l'échec, faces fautives). Taille bornée par
  `llm.max_attempts_history` et `llm.max_faces` : mesurée entre 150 et 3 300 tokens sur les
  pièces d'essai, soit **`-c 8192` largement suffisant**.
- **Sortie** : un JSON contraint par schéma, une action parmi `alg`, `free`, `merge`,
  `size`, `subdiv`, `microfix`, `tet`, plus une phrase de justification.
- **Le LLM ne construit jamais la recette** : il choisit un levier, et le code applique ce
  levier (`strategy/levers.py`), filtre les faces inventées et borne les valeurs. Une action
  inconnue, un serveur absent, un délai dépassé ou un JSON illisible font simplement
  repartir sur les règles déterministes — le maillage n'échoue jamais à cause du LLM.
- **Traçabilité** : chaque décision (action, recette obtenue, raison, durée, tokens estimés,
  réponse brute) est écrite dans le `.json` sous la clé `llm`.

| Clé `[llm]` | Défaut | Rôle |
|---|---|---|
| `enabled` | false | activer le conseiller |
| `base_url` | http://127.0.0.1:8080 | adresse de llama-server |
| `timeout_s` / `max_tokens` | 60 / 256 | délai d'une décision, longueur de la réponse |
| `max_attempts_history` / `max_faces` | 10 / 12 | bornes de la charge utile |
| `min_attempts_before` | 1 | essais échoués avant de solliciter le LLM |

## 8. Configuration

`stepmesher dump-config` affiche tous les paramètres commentés. Les principaux :

| Clé | Défaut | Rôle |
|---|---|---|
| `mesh.size_frac` | 0,01 | taille cible = fraction de la diagonale OBB… |
| `mesh.thickness_factor` | 10 | … plafonnée à k × épaisseur locale… |
| `mesh.min_size_mm` / `max_size_mm` | 0,5 / 50 | … et bornée |
| `mesh.n_per_bend` / `n_per_hole` | 4 / 12 | raffinement des plis / trous conservés |
| `mesh.micro_edge_ratio` | 0,2 | bord < ratio × taille mini = imposé par la CAD |
| `mesh.adaptive_attempts` | 8 | essais adaptatifs supplémentaires par passe |
| `holes.max_diameter_mm` / `max_diameter_frac` | 12 / 0,02 | seuil de bouchage = min des deux |
| `healing.small_edge_tol_mm` | [0.02, 0.01] | tolérances de la passe B |
| `quality.soft_violation_pct` | 0,2 | % d'éléments tolérés hors cible |
| `tet.order` | 2 | 2 = C3D10, 1 = C3D4 |
| `general.attempt_timeout_s` / `part_time_budget_s` | 600 / 2400 | délais |

## 9. Tests et résultats

```bash
pytest -m "not real"     # pièces synthétiques
pytest -m real           # pièces réelles de stps_test/ (lent)
```

Couverture : épaisseurs à ±1 %, nature, plis (rayon et angle exacts), trous bouchés et
conservés, arêtes vives, choix de la peau de référence, paliers d'épaisseur, 100 % quads,
1 élément dans l'épaisseur, jacobien, onglet au coin trièdre, reprojection < 1 %, bloc massif
→ C3D10 (ordre des nœuds milieux), validateur `.inp`, aller-retour `.inp`, ASCII pur,
orthonormalité des repères, invariance du hash par déplacement rigide, confinement des
plantages et délais, garde-fou de performance.

État : **92 tests verts** (85 synthétiques en ~4 min, dont 12 sur le conseiller LLM avec un
faux llama-server, et 7 sur pièces réelles en ~20 min).

### Pièces réelles `stps_test/` (conteneur 2 cœurs / 7 Go)

| Pièce | Nature détectée | Statut | Éléments | Qualité min / 5e centile | Hors cible | Essais | Temps |
|---|---|---|---|---|---|---|---|
| part_000 | tôle constante 4,34 mm, 3 plis | **OK** | 4 513 SC8R | SJ 0,163 / 0,82 | 0,09 % | 1 | 7 s |
| part_093 | usinée 3 / 4,3 / 6 / 8,3 mm, jonction en T | **OK_TET** | 209 834 C3D10 | forme 0,002* / 0,49 | – | 24 | 346 s |
| part_145 | variable, 6 paliers 3,4 → 10 mm | **OK_APPROX** | 17 344 SC8R | SJ 0,158 / 0,71 | 0,18 % | 3 | 182 s |
| part_201 | variable 3,2 / 5,9 / 8,3 mm | **OK_APPROX** | 12 814 SC8R | SJ 0,418 / 0,87 | 0 % | 1 | 104 s |
| part_349 | variable 1,14 / 1,24 / 1,99 mm, 4 plis | **OK_APPROX** | 67 320 SC8R | SJ 0,106 / 0,73 | 0,19 % | 2 | 417 s |

SJ = jacobien normalisé. Tous les `.inp` passent le validateur interne. Toutes les pièces
passent en géométrie brute (passe A).
\* part_093 : 33 tétras de forme < 0,05 exemptés car imposés par des micro-arêtes CAD
(arête < 10 % de la taille mini) ; écart de volume maillage / CAD 0,5 %. Les 24 essais SC8R
(passes A et B) échouent sur des éléments retournés à la jonction en T avant le repli.

## 10. Limites connues

- **Épaisseur variable** : maillage approché (`OK_APPROX`). Chaque nœud est décalé jusqu'à la
  peau opposée locale, mais les marches d'épaisseur ne sont pas alignées sur le maillage : les
  éléments à cheval forment une rampe (comptés dans `n_step_elements`). Une `*SHELL SECTION`
  par palier.
- **part_093** (jonction en T / nervures) part en tétra : une peau décalée ne peut pas la
  représenter en SC8R.
- **Mémoire de recettes** : l'empreinte (hash exact invariant + vecteur de caractéristiques)
  est calculée et stockée dans le JSON ; la base SQLite et la réutilisation ne sont pas encore
  branchées (statut mémoire toujours `NEW`). LLM optionnel : non fait.
- **Temps** : les grosses pièces à épaisseur variable demandent plusieurs essais
  (3 à 5 min par essai sur part_349 avec 2 cœurs).
- **Déterminisme** : avec `threads > 1`, gmsh peut produire quelques éléments de plus ou de
  moins d'une exécution à l'autre. `threads = 1` pour un résultat reproductible.
- **Stratégie `compound`** (surfaces composites globales) : disponible mais retirée des défauts
  (plantages et délais sur les pièces CATIA testées) ; la fusion locale reste un levier adaptatif.
- **À confirmer au premier datacheck Abaqus** :
  1. syntaxe de l'orientation par `*DISTRIBUTION` ;
  2. pour les continuum shells, l'épaisseur mécanique vient de la géométrie nodale ; la valeur
     écrite dans `*SHELL SECTION` est l'épaisseur nominale de la zone ;
  3. `STACK DIRECTION=3` avec la numérotation 1-4 / 5-8 produite ;
  4. C3D10 : ordre des nœuds milieux (vérifié par géométrie, à confirmer par Abaqus).
