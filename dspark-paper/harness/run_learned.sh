#!/bin/zsh
# The stages a build that changes the scheduler must re-measure: every arm that runs AdaSpark's cost
# model, and the width table. Everything else stays -- the llama.cpp arms, the autoregressive arms,
# the chains and the fixed tree never read the cost model.
#
#   per model, the width table first (all models), then the rest:
#     {m}_learned          agentic.py on the evaluation subset: AdaSpark, the same scheduler with its
#                          width pinned to 8, 12 and 16 rows (complete-wN: the width table), and on
#                          the LFM2.5 pair the cost-model ablation arm
#     {m}_learned_rep      AdaSpark again on the evaluation subset, after the pinned arms: its own
#                          run-to-run spread, and a bracket around the pinned arms in time
#     moekm_learned_narrow (+ _rep) the routed target only: AdaSpark and widths pinned to 4, 5 and 6
#                          rows, bracketed the same way. Its verify time rises with every row, so a
#                          width under 8 can win there; on the dense targets 2 to 8 rows cost within
#                          4% of 8 (the cost survey), so a narrower pinned width only loses tokens
#     {m}_figs_learned     the probed run of AdaSpark and the cost-model arm (figures)
#     {m}_heldout_learned  AdaSpark on the held-out conversations (llama.cpp's rows stay)
#     {m}_toolace_learned  the cost-model arms on the ToolACE conversations (the other arms' rows stay)
#
# usage: run_learned.sh OUTDIR [MODEL ...]   MODEL in lfm26 moekm q4b q8b (default: all four)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run_learned.sh OUTDIR [MODEL ...]}
shift
if (( $# )); then MODELS=($@); else MODELS=(q4b moekm lfm26 q8b); fi
MAXTOK=8192
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf
WIDTHS=complete-w8,complete-w12,complete-w16
NARROW=complete-w4,complete-w5,complete-w6

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN_LEARNED start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- crates | wc -l | tr -d ' ') server=$(md5 -q target/release/imparo-server) forward=$(md5 -q target/release/imparo-forward) models=${MODELS[*]}"

agentic() {  # tag arms set-flag subset-flag env...
  local tag=$1 arms=$2 set=$3 subset=$4
  shift 4
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q target/release/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arms 1 $MAXTOK $set $subset > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN_LEARNED STOPPED at $tag"; exit 1; }
}

figs_env=(IMPARO_DSPARK_ROUND=1 IMPARO_DSPARK_SCORE=1 IMPARO_DSPARK_TRACE=1)

env_of() {  # model -> its cost-model arms, then its environment
  case $1 in
    lfm26) print -r -- "budget,complete" AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) print -r -- "budget,complete" AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   print -r -- "complete" AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    q8b)   print -r -- "complete" AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-8B-GGUF/Qwen3-8B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $1"; exit 1 ;;
  esac
}

for m in $MODELS; do
  spec=(${(z)$(env_of $m)})
  arms=${spec[1]}; envs=(${(Q)spec[2,-1]})
  agentic ${m}_learned "$arms,$WIDTHS" --set=public --subset=core "${envs[@]}"
  agentic ${m}_learned_rep complete --set=public --subset=core "${envs[@]}"
  if [ $m = moekm ]; then
    agentic ${m}_learned_narrow "complete,$NARROW" --set=public --subset=core "${envs[@]}"
    agentic ${m}_learned_narrow_rep complete --set=public --subset=core "${envs[@]}"
  fi
done
for m in $MODELS; do
  spec=(${(z)$(env_of $m)})
  arms=${spec[1]}; envs=(${(Q)spec[2,-1]})
  agentic ${m}_figs_learned complete,budget --set=public --subset=smoke "${envs[@]}" $figs_env
  agentic ${m}_heldout_learned complete --set=public --subset=rest "${envs[@]}"
  agentic ${m}_toolace_learned $arms --set=toolace "" "${envs[@]}"
done
say "RUN_LEARNED DONE $(date '+%m-%d %H:%M:%S')"
