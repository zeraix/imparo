#!/bin/zsh
# Proves a build that routes chain rounds through the tree verify changed nothing but time.
# Short subset, one server per case; every reply's text hash is compared with the campaign's run
# of the same arm on the previous build.
#
#   chain3 on the tree route  (default)             == old chain3   and faster where context is long
#   chain3 on the block route (CHAIN_ROUTE=block)   == old chain3   and the old speed
#   complete (tree path, untouched)                 == old complete and the old speed
#
# usage: verify_route.sh RUNDIR          (RUNDIR holds the campaign's {m}_smoke stages)
set -u
cd "$(dirname "$0")/../.."
H=dspark-paper/harness
RUN=${1:?usage: verify_route.sh RUNDIR}
OUT=$RUN/verify_route
mkdir -p $OUT
export AGENTIC_CTX=32768
LFM=models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf
LFM_D=models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf
Q4=models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
Q4_D=models/Qwen3-4B-GGUF/Qwen3-4B-DSpark-Q4_K_M.gguf
echo "VERIFY start $(date '+%m-%d %H:%M:%S') server=$(md5 -q target/release/imparo-server)" | tee $OUT/VERIFY.log

one() {  # tag arm old-stage env...
  local tag=$1 arm=$2 old=$3
  shift 3
  env "$@" python3 -u $H/agentic.py $OUT/$tag $arm 1 8192 --set=public --subset=smoke > $OUT/$tag.log 2>&1 \
    || { echo "FAILED $tag (see $OUT/$tag.log)" | tee -a $OUT/VERIFY.log; exit 1; }
  echo "== $tag vs $old" | tee -a $OUT/VERIFY.log
  python3 $H/verify_route.py $RUN/$old/results.jsonl $arm $OUT/$tag/results.jsonl $arm | tail -1 | tee -a $OUT/VERIFY.log
}

one lfm26_chain3_tree  chain3   lfm26_smoke AGENTIC_TARGET="$LFM" AGENTIC_DRAFT=$LFM_D
one lfm26_chain3_block chain3   lfm26_smoke AGENTIC_TARGET="$LFM" AGENTIC_DRAFT=$LFM_D IMPARO_DSPARK_CHAIN_ROUTE=block
one lfm26_complete     complete lfm26_smoke AGENTIC_TARGET="$LFM" AGENTIC_DRAFT=$LFM_D
one q4b_chain3_tree    chain3   q4b_smoke   AGENTIC_ENABLE_THINKING=1 AGENTIC_TARGET=$Q4 AGENTIC_DRAFT=$Q4_D
echo "VERIFY done $(date '+%m-%d %H:%M:%S')" | tee -a $OUT/VERIFY.log
