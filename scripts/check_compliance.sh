#!/usr/bin/env bash
# 1) no forbidden modules anywhere in our code  2) starter files are byte-identical to the originals
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
pattern='nn\.Transformer|TransformerEncoder|TransformerDecoder|MultiheadAttention|scaled_dot_product_attention|^\s*(from|import)\s+transformers|from_pretrained'
hits=$(grep -rnE "$pattern" --include='*.py' --include='*.ipynb' \
       --exclude-dir=WikiSQL --exclude-dir=.git --exclude-dir=scripts \
       --exclude-dir=.venv --exclude-dir=venv --exclude-dir=env --exclude-dir=__pycache__ . || true)
if [ -n "$hits" ]; then echo "FORBIDDEN API FOUND:"; echo "$hits"; status=1
else echo "forbidden-API scan: clean"; fi
if (cd starter && sha256sum --quiet -c ../scripts/starter.sha256); then echo "starter/: unchanged"
else echo "starter/: MODIFIED"; status=1; fi
exit $status
