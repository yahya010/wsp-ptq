"""Likelihood side of the test panel (transformers, one GPU), run once at the end: likelihood_test.py <tag> [edit.pt ...]
WikiText-2 test perplexity (1,024-token windows) and MMLU test (14,042 questions, 0-shot letter likelihood).
<tag> = "ref" for the reference itself; otherwise the start checkpoint plus the listed edits."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import w3  # noqa: E402
from lab import DATA, Lab  # noqa: E402

torch.set_grad_enabled(False)


def main():
    tag, edits = sys.argv[1], sys.argv[2:]
    lab = Lab(n_math=8, n_code=4, n_prose=4)
    lab.load_mmlu(DATA / "mmlu_test.json")
    if tag != "ref":
        lab.set_state(w3.overlay(w3.load(), *edits))
    ids = lab.tok(open(DATA / "wikitext2_test.txt").read()).input_ids
    nll, n = 0.0, 0
    for i in range(0, len(ids) - 1023, 1024):
        x = torch.tensor([ids[i:i + 1024]], device=lab.dev); h = lab.bb(input_ids=x).last_hidden_state[0][:-1]
        nll -= float(torch.log_softmax(lab.head(h).float(), -1).gather(1, x[0, 1:, None]).sum()); n += 1023
    acc, _ = lab.mmlu()
    res = {"tag": tag, "wikitext2_ppl": round(float(torch.exp(torch.tensor(nll / n))), 4), "mmlu": round(acc, 4)}
    from common import write
    write(f"likelihood-test-{tag}", res, [])


if __name__ == "__main__":
    main()
