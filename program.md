# wsp-ptq

You are repairing a quantized language model by reading and editing its weights directly ("weight-space programming").
You work alone and in a loop: measure, find what is broken, understand why, edit, validate, keep or revert, repeat.

## The task

> This 3-bit checkpoint should behave like the NVFP4 model. Find what is broken and fix it inside the 3-bit format.
> You may run the model and read its weights and activations; you may not fit weights to data (no GPTQ, no
> least-squares refits, no training).

- **The model:** NVIDIA Nemotron-3.5-Lightning-30B-A3B, a hybrid Mamba / attention / mixture-of-experts model.
- **The reference** (`work/ref`): NVIDIA's NVFP4 release, dequantized to BF16. It is the target. "Behave like" means
  its output distribution: KL(reference || yours) → 0, and its benchmark scores.
- **The start checkpoint** (`work/start`): the same model quantized to 3 bits in the W3 format. `work/start/wsp_start.json`
  says which one you were given. It is worse than the reference; finding out where, and why, is the task.

## Setup

1. Check that `prepare.py` has run: `work/ref`, `work/start` and `work/data` exist. If not, tell the human.
2. Read `README.md`, `w3.py` (the format, and the only code that reads or writes checkpoints), `lab.py` (validation
   metrics, swapping matrices, reading activations) and `evals/` (benchmarks). Do not modify these files, `prepare.py`,
   or anything in `work/data`.
3. Make a run directory `runs/<date>-<start checkpoint>/` and work only there: your analysis scripts, `edits/`,
   `results.tsv`, `notes.md`.
4. Measure the baseline: `lab.full()` with the start checkpoint swapped in, and `evals/dev.sh start` (no edits). Log
   both as row 0. Also check that `lab.full()` on the untouched reference gives KL 0.

## The format, and what you may change

A quantized matrix is stored as 3-bit codes into an 8-level codebook, one E4M3 (FP8) scale per group of 32 consecutive
weights of a row, and one fp32 scale per row: `W[i, j] = levels[codes[i, j]] * scale[i, j // 32] * row[i]`. Torch
layout: rows are output features, columns are input features, so a group is 32 adjacent inputs of one output. See
`w3.py` for the exact definition.

You may change anything the format can represent: codes, group scales, row scales, the codebook of a matrix. The
result must be a valid W3 checkpoint with the same layout, size and kernels. No extra tensors, no higher-precision
weights, no matrices taken out of quantization.

## Rules

1. **No fitting.** Every value you write must follow from a rule you can state in one or two sentences, using only
   the checkpoint, the reference weights and the format. Not allowed: GPTQ or any other error-compensating quantizer,
   least-squares or closed-form refits against activations, gradient steps, training, distillation, or searching over
   values and keeping whichever scores best on data.
2. **Data finds problems and judges edits; it does not choose values.** You may run the model and read activations to
   find where to look and to understand what you see. Validation decides keep or revert for a whole edit. If you find
   yourself tuning a knob over many values by validation score, that is fitting: stop.
3. **Activation-dependent edits are flagged.** If an edit's values depend on activation statistics at all (for
   example, weighting rounding error by input magnitude), mark it `uses_activations=yes`, keep it a separate edit,
   and say so in the final report so the results can be given with and without it. If a human is around, ask first.
4. **Validation only, test once.** Make every decision on validation: `lab.full()` (KL on chat math / code / prose,
   WikiText-2 validation perplexity, MMLU validation) and `evals/dev.sh` (vLLM NLL of the reference's answers, GSM8K
   dev1k). Run `evals/test_panel.sh` once, on the final checkpoint, after every decision is made. Never look at test
   sets before then.
5. **One hypothesis per edit.** Test edits one at a time on top of the current best checkpoint. Do not batch several
   untested changes into one edit.
6. **Count what you change.** Report `w3.changed(start, current)` (codes, group scales, share of all quantized weights)
   for every kept edit.

## The loop

1. **Localize.** Find where the remaining error comes from: which layers, which matrices, which tokens. The lab can put
   any matrix back to its reference value (`lab.restore([...])`), give per-token KL (`lab.kl(group, per_token=True)`)
   and capture the activations entering a matrix (`lab.inputs`). `w3.reference(name)` gives any matrix's reference
   weights. How you localize is up to you.
2. **Read.** Look at the worst unit concretely: print its weights, codes, scales, the reference weights it should
   match, and the activations that multiply it. Compare what is stored with what the format could have stored.
3. **Explain.** Write down, in `notes.md`, in a few sentences: what is wrong, the evidence (numbers), why the
   quantizer produced it, and what change inside the format would fix it. If you cannot explain it, read more before
   editing.
4. **Edit.** Write the fix as a script that saves only the changed matrices to `edits/<id>.pt` (a `{name: matrix}`
   dict in the `w3` state format). Edits apply on top of the start checkpoint with `w3.overlay(w3.load(), *edits)`.
5. **Validate.** Run `lab.full()` and `evals/dev.sh <id> <edits...>` with the kept edits so far plus the new one.
   Keep the edit only if validation improves where it matters (chat-math KL, vLLM NLL, GSM8K dev1k) without clearly
   hurting the rest. The metrics disagree sometimes, and GSM8K dev1k moves by about a point between equivalent
   checkpoints: never decide on one small difference in one metric.
6. **Log.** Append a row to `results.tsv` and the reasoning to `notes.md`, whether kept or reverted.
7. **Repeat** from step 1 on the new best checkpoint. A fix can expose a problem that was hidden before it.

Stop when no candidate edit improves validation. Then run the test panel once and write the report.

## Logging

`results.tsv`, tab-separated, one row per edit tried:

```
id  parent  matrices  codes_changed  scales_changed  uses_activations  math_kl  code_kl  prose_kl  ppl  mmlu_val  nll  gsm8k_dev1k  status  description
```

`status` is keep, revert or crash. `description` is one line; the full reasoning goes in `notes.md`.

## Final report (`runs/<run>/REPORT.md`)

- The kept edits, in order. For each: what was wrong, the evidence, the rule that set the new values, codes and scales
  changed, and its effect on validation.
- One table, start vs final, on validation and on the test panel, with the share of weights changed.
- Every flagged (activation-dependent) edit, and the results without it.
- What did not work, briefly.

## Practical notes

- One H100 80 GB holds the BF16 lab (about 60 GB) or one vLLM instance, not both. The evals already set the vLLM
  concurrency that fits on one GPU.
- Keep `TP=1` for every vLLM number you compare: on quantized checkpoints greedy GSM8K changes by several points
  between TP=1 and TP=2.
- transformers runs NemotronH's Mamba layers on a slow PyTorch path that runs out of memory on sequences of 2,048+
  tokens and large batches; the lab uses 1,024-token windows.
- A BF16 copy for vLLM is about 60 GB. Delete it after its evals; it can be rebuilt from the edits.
- Killing vLLM can leave an orphan `VLLM::EngineCore` process holding the GPU. Kill processes by PID; never `pkill -f`
  a pattern that also matches your own command line.
- A full state (`w3.load()`) takes about 32 GB of CPU memory. Load only the matrices you need with `w3.load(names=[...])`
  while exploring.
