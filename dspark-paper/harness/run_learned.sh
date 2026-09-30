#!/bin/zsh
# The stages a build that changes only how the cost model forms its width classes must re-measure:
# every arm that runs the cost model. Everything else run.sh and run_chain.sh measured stays -- the
# llama.cpp arms, the autoregressive arms, the chains and the fixed tree never read the cost model.
#
#   per model: {m}_learned         agentic.py on the evaluation subset: AdaSpark, and on the LFM2.5
#                                  pair the two ablation arms that run the cost model
#              {m}_widths_learned  table4.py, the chosen-width column; the store is learned once per
#                                  model over all nine (context, prompt) cells before measuring
#              {m}_figs_learned    the probed run of AdaSpark and the cost-model arm (figures)
#
# usage: run_learned.sh OUTDIR [MODEL ...]   MODEL in lfm26 moekm q4b q8b (default: all four)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run_learned.sh OUTDIR [MODEL ...]}
shift
if (( $# )); then MODELS=($@); else MODELS=(lfm26 moekm q4b q8b); fi
MAXTOK=8192
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN_LEARNED start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- crates | wc -l | tr -d ' ') server=$(md5 -q target/release/imparo-server) forward=$(md5 -q target/release/imparo-forward) models=${MODELS[*]}"

agentic() {  # tag arms subset-flag env...
  local tag=$1 arms=$2 subset=$3
  shift 3
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q target/release/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arms 1 $MAXTOK --set=public $subset > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN_LEARNED STOPPED at $tag"; exit 1; }
}

step() {  # tag command...
  local tag=$1
  shift
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  "$@" > $OUT/$tag.log 2>&1
  local rc=$?
  say "EXIT $tag=$rc $(date '+%m-%d %H:%M:%S')"
  [ $rc = 0 ] || { tail -5 $OUT/$tag.log | tee -a $OUT/RUN.log; say "RUN_LEARNED STOPPED at $tag"; exit 1; }
}

figs_env=(IMPARO_DSPARK_ROUND=1 IMPARO_DSPARK_SCORE=1 IMPARO_DSPARK_TRACE=1)

model() {  # name arms prompt-set env...
  local m=$1 arms=$2 pset=$3
  shift 3
  agentic ${m}_learned $arms --subset=core "$@"
  local tgt draft
  for kv in "$@"; do
    case $kv in AGENTIC_TARGET=*) tgt=${kv#*=};; AGENTIC_DRAFT=*) draft=${kv#*=};; esac
  done
  step ${m}_widths_learned env T4_WARM=model SURVEY_MODEL="$tgt" SURVEY_DRAFT="$draft" PROMPT_SET=$pset \
    python3 -u $H/table4.py $OUT/${m}_widths_learned budget 443,1596,8444 A,B,C 256 2
  agentic ${m}_figs_learned complete,budget --subset=smoke "$@" $figs_env
}

for m in $MODELS; do
  case $m in
    lfm26) model lfm26 budget,complete lfm25 \
             AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) model moekm budget,complete lfm25 \
             AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   model q4b complete qwen3 AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    q8b)   model q8b complete qwen3 AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-8B-GGUF/Qwen3-8B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $m"; exit 1 ;;
  esac
done
say "RUN_LEARNED DONE $(date '+%m-%d %H:%M:%S')"
