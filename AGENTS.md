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

pytest -m "not real"             # 85 tests synthétiques (~2-5 min)
pytest -m real                   # tests sur stp_verif/ (lents, ~20 min)
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
- `threads > 1` : gmsh non déterministe (± quelques dizaines d'éléments).
- Les `.inp` sont en ASCII pur (pas d'accents dans Abaqus).
- Nettoyer les fichiers/dossiers temporaires (`proto_*`, sorties de debug) après usage.

## Workflow attendu

1. Lire `CLAUDE.md` (+ la mémoire repo `/memories/repo/`) avant un nouveau chantier.
2. Prototyper isolément (script jetable) avant d'intégrer ; mesurer sur `stp_verif/`.
3. Après toute modif : `pytest -m "not real"` + relancer les pièces réelles impactées
   (non-régression : statuts et nb d'éléments stables).
4. Mettre à jour `CLAUDE.md` et la mémoire repo.

## Chantier en cours / prochain pas

**Fait** : outil de nettoyage STEP (OCP `ShapeFix_Wireframe`), intégré, systématique,
non-régression validée (5 pièces + 85 tests).

**Prochain pas — analyser la campagne complète de Fred** (il la lance maintenant sur
beaucoup de pièces, nettoyage OCP actif) :
- collecter les `.json` + `summary.csv`, classer les statuts et les causes d'échec restantes ;
- pour le nettoyage : compter `cleaned` / `unchanged` / `rejected` / `error`, distribution
  du nombre de micro-arêtes effondrées et des dV/V, examiner les rejets ;
- décider si ajuster `occ_wireframe_precision_mm` ou traiter d'autres profils de défauts
  (faces-slivers d'aire faible, micro-faces) — **le defeaturing est un no-op, ne pas y revenir**.
