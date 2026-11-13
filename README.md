# wsq

Repair a quantized LLM by letting an LLM agent read and edit its weights directly: weight-space programming
for post-training quantization.

You give an agent a 3-bit checkpoint that has lost accuracy, the higher-precision model it came from, and a small lab
for measuring the gap. The agent finds what is broken, explains why, and fixes it by editing codes and scales inside
the 3-bit format. It may run the model and read weights and activations, but it may not fit weights to data (no GPTQ,
no least-squares refits, no training). It works in a loop: measure, localize, read, explain, edit, validate, keep or
revert.

As in [autoresearch](https://github.com/karpathy/autoresearch), the part you edit is the prompt: **`program.md`**.
Point your agent at it.

Blog post: [Repairing a Quantized LLM using Weight Space Programming](https://sharar-project.github.io/blog/repairing-quantized-llm/)

## What one run did

On NVIDIA Nemotron-3.5-Lightning-30B-A3B, starting from a poorly calibrated 3-bit GPTQ checkpoint, the agent edited
0.0008% of the weights. We compare it with re-running GPTQ on the same validation data as a new calibration set. Test
sets, run once:

| Checkpoint | MATH-500 ↑ | LiveCodeBench v6 ↑ | GPQA Diamond ↑ | MMLU ↑ | WikiText-2 PPL ↓ |
| --- | --- | --- | --- | --- | --- |
| NVFP4 release (reference) | 88.0 | 55.6 | 72.1 | 77.0 | 8.08 |
| Poorly calibrated (GPTQ) | 61.6 | 30.0 | 46.6 | 66.6 | 11.76 |
| + re-calibration (GPTQ) | **84.4** | 35.6 | 62.6 | 68.5 | 11.18 |
| + agent edits (WSP) | 83.0 | **47.8** | **64.3** | **74.3** | **9.38** |

Bold marks the best 3-bit score in each column. MATH-500 uses greedy decoding with thinking off. LiveCodeBench v6
(131 problems from February 2025 on) and GPQA Diamond use thinking on and a 32K-token limit, averaged over 3 and 4
samples. MMLU is the 0-shot test split; WikiText-2 is perplexity on the test split. The validation data used for the
repair and the re-calibration is published as [wsp-agent-data](https://huggingface.co/datasets/yemara/wsp-agent-data).

What it found and how is in the blog post. `program.md` does not contain any of it: an agent starting from this repo
has to find the problems itself.

## Files

```
program.md        the agent's instructions: task, rules, loop, logging (edit this)
prepare.py        one-time setup: reference model, start checkpoint, validation and test data
w3.py             the 3-bit W3 format: load / save checkpoints, dequantize, apply edits, count changes
lab.py            validation lab (transformers): KL to the reference, perplexity, MMLU-val, swap matrices, read activations
build.py          start checkpoint + edits -> BF16 copy for vLLM
evals/dev.sh      validation in vLLM, for keep / revert: NLL of the reference's answers, GSM8K dev1k
evals/test_panel.sh   the test panel, run once at the end: GSM8K, MATH-500, LiveCodeBench v6, GPQA Diamond, WikiText-2, MMLU
runs/             the agent's working directories (one per run)
```

The agent does not modify `prepare.py`, `w3.py`, `lab.py`, `build.py` or `evals/`. It writes its own analysis
scripts and its edits (small files holding only the changed matrices) in `runs/<run>/`.

## Setup

**Hardware:** one 80 GB GPU (H100) is enough; two let the lab and vLLM run side by side. About 200 GB of disk.

**Software:** we used two Python 3.12 environments:
- **vLLM** (vLLM 0.30, torch 2.13, transformers 5.16) for `prepare.py`, `build.py` and `evals/`;
- **lab** (torch 2.11, transformers 5.17, accelerate) for `lab.py`. `evals/test_panel.sh` calls the lab environment
  through `LAB_PYTHON` for its likelihood part.

Both need `safetensors`, `huggingface_hub`, `pandas`, `pyarrow` and `numpy`; the vLLM one also `math-verify` for
MATH-500. LiveCodeBench needs a [checkout](https://github.com/LiveCodeBench/LiveCodeBench) at commit `28fef95`, with
`LCB_DIR` pointing to it. GPQA is gated: accept its terms on the Hub first.

```bash
python prepare.py ref                       # NVFP4 reference -> work/ref (BF16)
python prepare.py start --checkpoint poor   # or: well
python prepare.py data                      # validation + test sets -> work/data (runs the reference once in vLLM)
```

Start checkpoints: [poorly calibrated](https://huggingface.co/yemara/Nemotron-3.5-Lightning-30B-A3B-W3-PoorlyCalibrated),
[well-calibrated](https://huggingface.co/yemara/Nemotron-3.5-Lightning-30B-A3B-W3-WellCalibrated). Repaired by the
run above: [poorly calibrated + agent edits](https://huggingface.co/yemara/Nemotron-3.5-Lightning-30B-A3B-W3-WSP-PoorlyCalibrated),
[well-calibrated + agent edits](https://huggingface.co/yemara/Nemotron-3.5-Lightning-30B-A3B-W3-WSP-WellCalibrated).

## Running the agent

Start Claude Code, Codex or another coding agent in this directory and prompt something like:

```
Read program.md and let's start a new run. Do the setup first.
```

The agent measures the start checkpoint, then loops on its own. Each edit it tries becomes a row in
`runs/<run>/results.tsv` and its reasoning goes in `runs/<run>/notes.md`. At the end it runs the test panel once and
writes `runs/<run>/REPORT.md`.

The validation sets differ slightly from the ones behind the table above (the code set here is built from Python's
standard library), so validation numbers will not match the blog post exactly. The test sets are the same.

## License

MIT. See [LICENSE](LICENSE).
