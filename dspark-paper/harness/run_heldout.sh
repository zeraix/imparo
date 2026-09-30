#!/bin/zsh
# The held-out conversations: llama.cpp's DSpark and AdaSpark on the 31 conversations (61 turns) of
# the test set outside the evaluation subset (agentic.py --subset=rest), which no arm had run. If
# AdaSpark over llama.cpp's DSpark there matches the evaluation subset, the subset did not favour
# it; the two together cover all 148 turns.
#
# usage: run_heldout.sh OUTDIR [MODEL ...]   MODEL in lfm26 moekm q4b q8b (default: all four)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run_heldout.sh OUTDIR [MODEL ...]}
shift
if (( $# )); then MODELS=($@); else MODELS=(lfm26 moekm q4b q8b); fi
MAXTOK=8192
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN_HELDOUT start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- crates | wc -l | tr -d ' ') server=$(md5 -q target/release/imparo-server) models=${MODELS[*]}"

stage() {  # tag env...
  local tag=$1
  shift
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q target/release/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag llama-dspark,complete 1 $MAXTOK --set=public --subset=rest > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN_HELDOUT STOPPED at $tag"; exit 1; }
}

for m in $MODELS; do
  case $m in
    lfm26) stage lfm26_heldout AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) stage moekm_heldout AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   stage q4b_heldout AGENTIC_ENABLE_THINKING=1 AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    q8b)   stage q8b_heldout AGENTIC_ENABLE_THINKING=1 AGENTIC_TARGET=models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-8B-GGUF/Qwen3-8B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $m"; exit 1 ;;
  esac
done
say "RUN_HELDOUT DONE $(date '+%m-%d %H:%M:%S')"
