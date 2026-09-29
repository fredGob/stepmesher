#!/bin/bash
# Campagne en parallèle (Linux) : une pièce par processus, puis summary.csv reconstruit
# depuis les .json.
#
#   tools/run_campaign.sh <dossier_stp> <dossier_sortie> [parallélisme=8] [copie_src] [config.toml]
#
# copie_src : dossier contenant une COPIE FIGÉE de src/ (PYTHONPATH) ; les essais tournent en
# sous-processus qui réimportent le code : modifier src/ pendant un lot mélange deux versions.
#   cp -r src /tmp/vN_src && tools/run_campaign.sh campagne/parts_stp_echelle result_vN 10 /tmp/vN_src
set -u
SRC=$1; OUT=$2; P=${3:-8}; PP=${4:-}; CFG=${5:-}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY="$PROJ/.venv/bin/python"
mkdir -p "$OUT"
# numpy/OpenBLAS prend sinon tous les cœurs dans chaque processus
export PYTHONPATH=$PP OUT PY CFG OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
T0=$(date +%s)
ls "$SRC"/*.stp | xargs -P "$P" -I{} sh -c '
  f="$1"; b=$(basename "$f" .stp)
  extra=""; [ -n "$CFG" ] && extra="--config $CFG"
  "$PY" -m stepmesher.cli mesh "$f" -o "$OUT" $extra > "$OUT/_run_$b.txt" 2>&1
' _ {}
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
from stepmesher.process import summary_row, write_summary
out = Path(sys.argv[1])
rows = [summary_row(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(out.glob("*.json"))]
write_summary(rows, out / "summary.csv")
EOF
echo "campagne $SRC -> $OUT : $(( $(date +%s) - T0 )) s"
