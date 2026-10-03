#!/bin/zsh
# Every measurement the paper reports, staged model by model, on one frozen build.
#
#   per model:  smoke   EVERY arm on --subset=smoke (one conversation per source, two turns),
#                       the autoregressive arms included: their rounds are also the
#                       speedup-over-own-autoregressive numbers (Qwen3's replies make a full-set
#                       autoregressive arm hours long)
#               check   check.py; a FAIL stops the whole run here
#               full    the speculative arms on the core subset (the first conversation of every
#                       category, half of each single-category source: 87 of 148 turns)
#               check
#               widths  table4.py: chain / fixed 8, 12, 16 / chosen at 443, 1596, 8444 keys, every
#                       arm from one store learned over all nine (context, prompt) cells
#               survey  survey.py: verify ms by pinned width and context
#               figs    AdaSpark and the cost-model arm with the round, score and chooser probes
#                       on, smoke subset (rounds, not turns, are the unit; not timed results)
#
# usage: run.sh OUTDIR [MODEL ...]      MODEL in lfm26 moekm q4b q8b (default: all, in that order)
# env:   REPEATS (full-set repeats, default 1; a second repeat runs the arms in reverse order)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run.sh OUTDIR [MODEL ...]}
shift
# zsh does not split words: build the list explicitly
if (( $# )); then MODELS=($@); else MODELS=(lfm26 moekm q4b q8b); fi
REPEATS=${REPEATS:-1}
# A reply that reaches the cap is a repetition loop under greedy decoding (the longest
# finished reply on the smoke subset was 4,391 tokens); 8192 bounds what a loop costs.
MAXTOK=8192
mkdir -p $OUT
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- crates | wc -l | tr -d ' ') server=$(md5 -q target/release/imparo-server) forward=$(md5 -q target/release/imparo-forward) models=${MODELS[*]} repeats=$REPEATS max_tokens=$MAXTOK"

agentic() {  # tag arms repeats subset-flag env...
  local tag=$1 arms=$2 reps=$3 subset=$4
  shift 4
  # RESUME: a stage that already passed its check on these binaries is not run again
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q target/release/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arms $reps $MAXTOK --set=public $subset > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN STOPPED at $tag"; exit 1; }
}

step() {  # tag command...
  local tag=$1
  shift
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  "$@" > $OUT/$tag.log 2>&1
  local rc=$?
  say "EXIT $tag=$rc $(date '+%m-%d %H:%M:%S')"
  [ $rc = 0 ] || { tail -5 $OUT/$tag.log | tee -a $OUT/RUN.log; say "RUN STOPPED at $tag"; exit 1; }
}

figs_env=(IMPARO_DSPARK_ROUND=1 IMPARO_DSPARK_SCORE=1 IMPARO_DSPARK_TRACE=1)
CORE=llama-dspark,chain3,complete
LADDER=llama-dspark,chain3,chain,tree16,budget,complete

model() {  # name speculative-arms prompt-set env...
  local m=$1 arms=$2 pset=$3
  shift 3
  agentic ${m}_smoke llama-plain,plain,$arms 1 --subset=smoke "$@"
  agentic ${m}_full $arms $REPEATS --subset=core "$@"
  local tgt draft
  for kv in "$@"; do
    case $kv in AGENTIC_TARGET=*) tgt=${kv#*=};; AGENTIC_DRAFT=*) draft=${kv#*=};; esac
  done
  step ${m}_widths env T4_WARM=model SURVEY_MODEL="$tgt" SURVEY_DRAFT="$draft" PROMPT_SET=$pset \
    python3 -u $H/table4.py $OUT/${m}_widths chain,8,12,16,budget 443,1596,8444 A,B,C 256 2
  step ${m}_survey env SURVEY_MODEL="$tgt" SURVEY_DRAFT="$draft" PROMPT_SET=$pset \
    python3 -u $H/survey.py $OUT/${m}_survey 2,4,6,8,10,12,16,20,24,32,40,48 443,1596,8444 128 2
  agentic ${m}_figs complete,budget 1 --subset=smoke "$@" $figs_env
}

for m in $MODELS; do
  case $m in
    lfm26) model lfm26 $LADDER lfm25 \
             AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) model moekm $LADDER lfm25 \
             AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   model q4b $CORE qwen3 AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    q8b)   model q8b $CORE qwen3 AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-8B-GGUF/Qwen3-8B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $m"; exit 1 ;;
  esac
done
say "RUN DONE $(date '+%m-%d %H:%M:%S')"
