#!/bin/bash
# Validation for keep / revert: evals/dev.sh <tag> [edit.pt ...]
# Builds start + edits as a BF16 copy, then runs vLLM NLL on the reference's GSM8K answers and GSM8K dev1k (TP=1).
# Results go to work/results/results.jsonl. Delete the BF16 copy afterwards (~60 GB): it is rebuildable from the edits.
set -e
cd "$(dirname "$0")/.."
tag=$1; shift
W=${WSP_ROOT:-work}
[ -f $W/bf16/$tag/config.json ] || python build.py $W/bf16/$tag "$@"
TP=1 python evals/nll.py $W/bf16/$tag $tag
TP=1 python evals/gsm8k.py $W/bf16/$tag $tag dev1k
