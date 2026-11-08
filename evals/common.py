"""Shared settings for the vLLM benchmarks. Every number in the README was produced with these: one GPU (TP=1), BF16
copy of the checkpoint (w3.to_bf16), seed 0. Keep TP fixed across everything you compare: on quantized checkpoints
greedy GSM8K moves by several points between TP=1 and TP=2."""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import w3  # noqa: E402

RESULTS = w3.ROOT / "results"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def llm(model, max_model_len, util=0.9, max_seqs=None):
    from vllm import LLM
    seqs = int(os.environ.get("MAX_SEQS", max_seqs or 0)) or None   # NemotronH on one H100: the Mamba cache limits concurrency
    return LLM(model, dtype="bfloat16", max_model_len=max_model_len, seed=0, tensor_parallel_size=int(os.environ.get("TP", "1")),
               gpu_memory_utilization=float(os.environ.get("GPU_UTIL", util)), **({"max_num_seqs": seqs} if seqs else {}))


def hf_file(repo, pattern, repo_type="dataset"):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, repo_type=repo_type, allow_patterns=[pattern])
    return sorted(glob.glob(f"{d}/{pattern}"))[0]


def write(name, res, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump({**res, "rows": rows}, open(RESULTS / f"{name}.json", "w"))
    with open(RESULTS / "results.jsonl", "a") as f:
        f.write(json.dumps({"name": name, **res}) + "\n")
    print("RESULT", json.dumps(res), flush=True)
