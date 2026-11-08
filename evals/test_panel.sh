#!/bin/bash
# The test panel. Run it ONCE per final checkpoint, after every keep / revert decision is made:
#   evals/test_panel.sh <tag> [edit.pt ...]        (tag "ref" with no edits evaluates the reference itself)
# GSM8K test, MATH-500, LiveCodeBench v6 (needs LCB_DIR), GPQA Diamond, then WikiText-2 test PPL and MMLU test
# (transformers; LAB_PYTHON = the python of the lab environment if it is not this one).
set -e
cd "$(dirname "$0")/.."
tag=$1; shift
W=${WSP_ROOT:-work}
if [ "$tag" = ref ]; then M=$W/ref; else M=$W/bf16/$tag; [ -f $M/config.json ] || python build.py $M "$@"; fi
export TP=1
python evals/gsm8k.py $M $tag test
python evals/math500.py $M $tag
python evals/lcb.py $M $tag
python evals/gpqa.py $M $tag
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${LAB_PYTHON:-python} evals/likelihood_test.py $tag "$@"
