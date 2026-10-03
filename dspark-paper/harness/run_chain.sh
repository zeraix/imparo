#!/bin/zsh
# The stages a build that changes only the chain verify route must re-measure. Everything else
# run.sh measured stays: its arms never execute the chain route (checked by verify_route.sh before
# this runs).
#
#   per model: {m}_chain         agentic.py, the chain arms on the evaluation subset
#                                (chain3 on every target; chain, the full block, on the LFM2.5 pair)
#
# usage: run_chain.sh OUTDIR [MODEL ...]   MODEL in lfm26 moekm q4b (default: all three)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run_chain.sh OUTDIR [MODEL ...]}
shift
if (( $# )); then MODELS=($@); else MODELS=(lfm26 moekm q4b); fi
MAXTOK=8192
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN_CHAIN start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- crates | wc -l | tr -d ' ') server=$(md5 -q target/release/imparo-server) forward=$(md5 -q target/release/imparo-forward) models=${MODELS[*]}"

agentic() {  # tag arms env...
  local tag=$1 arms=$2
  shift 2
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q target/release/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arms 1 $MAXTOK --set=public --subset=core > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN_CHAIN STOPPED at $tag"; exit 1; }
}


model() {  # name chain-arms prompt-set env...
  local m=$1 arms=$2 pset=$3
  shift 3
  agentic ${m}_chain $arms "$@"
}

for m in $MODELS; do
  case $m in
    lfm26) model lfm26 chain3,chain lfm25 \
             AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) model moekm chain3,chain lfm25 \
             AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   model q4b chain3 qwen3 AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $m"; exit 1 ;;
  esac
done
say "RUN_CHAIN DONE $(date '+%m-%d %H:%M:%S')"
