"""Teacher-forced NLL of the reference's own GSM8K answers (256 train questions), inside vLLM: nll.py <model dir> <tag>.
A cheap, low-noise validation number that agrees with how vLLM (not transformers) runs the model. Lower is better; the
reference itself scores lowest."""
import json
import sys

from common import llm, write
import w3


def main():
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    model, tag = sys.argv[1:3]
    rows = [r for r in json.load(open(w3.ROOT / "data" / "gsm8k_klgen_ref.json"))["rows"] if r["tokens"] < 2048][:256]
    m = llm(model, 4096, util=0.9, max_seqs=64)
    outs = m.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["output_ids"]) for r in rows],
                      SamplingParams(max_tokens=1, prompt_logprobs=0))
    nll, n, top = 0.0, 0, 0
    for r, o in zip(rows, outs):
        ids = r["prompt_ids"] + r["output_ids"]
        for t in range(len(r["prompt_ids"]), len(ids)):
            lp = o.prompt_logprobs[t][ids[t]]; nll -= lp.logprob; n += 1; top += lp.rank == 1
    write(f"nll-{tag}", {"tag": tag, "tokens": n, "nll": round(nll / n, 5), "top1": round(top / n, 4)}, [])


if __name__ == "__main__":
    main()
