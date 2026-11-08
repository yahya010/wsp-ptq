"""MATH-500 (test), 0-shot CoT, greedy, thinking off, max 4,096 new tokens, graded with math_verify: math500.py <model dir> <tag>"""
import json
import sys
import time

from common import SUFFIX, hf_file, llm, write


def main():
    from math_verify import parse, verify
    from vllm import SamplingParams
    model, tag = sys.argv[1:3]
    rows = [json.loads(l) for l in open(hf_file("HuggingFaceH4/MATH-500", "test.jsonl"))]
    m = llm(model, 6144, util=0.95, max_seqs=256)
    tok = m.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    t = time.time(); outs = m.generate(prompts, SamplingParams(temperature=0, max_tokens=4096)); dt = time.time() - t
    out, correct = [], 0
    for o, r in zip(outs, rows):
        text = o.outputs[0].text
        try:
            ok = bool(verify(parse("$" + r["answer"] + "$"), parse(text)))
        except Exception:
            ok = False
        correct += ok; out.append({"unique_id": r["unique_id"], "ok": ok, "tokens": len(o.outputs[0].token_ids), "text": text})
    write(f"math500-{tag}", {"tag": tag, "n": len(rows), "accuracy": correct / len(rows), "seconds": round(dt, 1),
                             "truncated": sum(x["tokens"] >= 4096 for x in out)}, out)


if __name__ == "__main__":
    main()
