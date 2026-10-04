#!/bin/zsh
# Table 9: learned widths against the 8-bit matrix kernel's row classes, on the evaluation subset.
# The kernel classes are not part of the released engine, so this runs a MEASUREMENT BUILD: the
# engine with harness/kernel_classes.patch applied (it restores the declared classes), built into
# its own target directory:
#   git worktree add ../imparo-classes && cd ../imparo-classes
#   git apply dspark-paper/harness/kernel_classes.patch && cargo build --release -p imparo-server
# There IMPARO_DSPARK_CLASSES=declared (agentic.py's arm complete-classes) prices the kernel's
# classes; without it the build learns its widths exactly as the released engine does.
#
#   per model:
#     {m}_classes      AdaSpark (learned widths) and AdaSpark on the kernel classes, one server each
#     {m}_classes_rep  AdaSpark again after them, so each comparison brackets the classes arm in time
#
# usage: run_classes.sh OUTDIR BINDIR [MODEL ...]   BINDIR holds the measurement build's imparo-server
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
OUT=${1:?usage: run_classes.sh OUTDIR BINDIR [MODEL ...]}
BIN=${2:?usage: run_classes.sh OUTDIR BINDIR [MODEL ...]}
shift 2
if (( $# )); then MODELS=($@); else MODELS=(q4b moekm lfm26 q8b); fi
MAXTOK=8192
export AGENTIC_CTX=32768 IMPARO_BIN_DIR=$BIN
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf

say() { echo "$*" | tee -a $OUT/RUN.log; }
say "RUN_CLASSES start $(date '+%m-%d %H:%M:%S') commit=$(git rev-parse --short HEAD) server=$(md5 -q $BIN/imparo-server) models=${MODELS[*]}"

agentic() {  # tag arms env...
  local tag=$1 arms=$2
  shift 2
  if grep -q '^CHECK PASS' $OUT/$tag.check 2>/dev/null && grep -q "imparo_md5=$(md5 -q $BIN/imparo-server)" $OUT/$tag.log 2>/dev/null; then
    say "=== $tag kept (passed earlier on this binary)"; return 0
  fi
  say "=== $tag start $(date '+%m-%d %H:%M:%S')"
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arms 1 $MAXTOK --set=public --subset=core > $OUT/$tag.log 2>&1
  local rc=$?
  python3 $H/check.py $OUT/$tag $OUT/$tag.log > $OUT/$tag.check 2>&1
  local ck=$?
  say "EXIT $tag=$rc check=$( [ $ck = 0 ] && echo PASS || echo FAIL ) $(date '+%m-%d %H:%M:%S')"
  grep '^NOTE' $OUT/$tag.check | tee -a $OUT/RUN.log
  [ $rc = 0 ] && [ $ck = 0 ] || { cat $OUT/$tag.check | tee -a $OUT/RUN.log; say "RUN_CLASSES STOPPED at $tag"; exit 1; }
}

env_of() {  # model -> its environment
  case $1 in
    lfm26) print -r -- AGENTIC_TARGET="$LFM" \
             AGENTIC_DRAFT=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf ;;
    moekm) print -r -- AGENTIC_TARGET=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf ;;
    q4b)   print -r -- AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf ;;
    q8b)   print -r -- AGENTIC_ENABLE_THINKING=1 \
             AGENTIC_TARGET=models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
             AGENTIC_DRAFT=models/Qwen3-8B-GGUF/Qwen3-8B-DSpark-Q4_K_M.gguf ;;
    *) say "unknown model $1"; exit 1 ;;
  esac
}

for m in $MODELS; do
  envs=(${(Q)${(z)$(env_of $m)}})
  agentic ${m}_classes complete,complete-classes "${envs[@]}"
  agentic ${m}_classes_rep complete "${envs[@]}"
done
say "RUN_CLASSES DONE $(date '+%m-%d %H:%M:%S')"
