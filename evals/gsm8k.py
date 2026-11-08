"""GSM8K, 0-shot CoT, greedy, thinking off, max 2,048 new tokens: gsm8k.py <model dir> <tag> <split>
splits: dev1k = 1,000 train questions (validation: use this to keep or revert edits); klgen = 256 other train questions
(the reference's answers become the chat-math KL set, prepare.py); test = the full 1,319-question test set (run once)."""
import re
import sys
import time

import numpy as np
import pandas as pd
from common import SUFFIX, hf_file, llm, write


def questions(split):
    if split == "test":
        return pd.read_parquet(hf_file("openai/gsm8k", "main/test-*.parquet"))
    df = pd.read_parquet(hf_file("openai/gsm8k", "main/train-*.parquet"))
    idx = np.random.default_rng(0).permutation(len(df))
    return df.iloc[{"klgen": idx[200:456], "dev1k": idx[1480:2480]}[split]].reset_index(drop=True)


def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", "").strip())
    return float(m[-1]) if m else None


def main():
    from vllm import SamplingParams
    model, tag, split = sys.argv[1:4]
    df = questions(split)
    m = llm(model, 4096, util=0.95, max_seqs=320)
    tok = m.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for q in df.question]
    t = time.time(); outs = m.generate(prompts, SamplingParams(temperature=0, max_tokens=2048)); dt = time.time() - t
    rows, correct = [], 0
    for o, a, q in zip(outs, df.answer, df.question):
        text = o.outputs[0].text
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        pred, gold = (num(boxed[-1]) if boxed else num(text)), num(a.split("####")[-1])
        ok = pred is not None and abs(pred - gold) < 1e-6; correct += ok
        rows.append({"question": q, "gold": gold, "pred": pred, "ok": ok, "tokens": len(o.outputs[0].token_ids), "text": text,
                     "prompt_ids": list(o.prompt_token_ids), "output_ids": list(o.outputs[0].token_ids)})
    write(f"gsm8k-{split}-{tag}", {"tag": tag, "split": split, "n": len(df), "accuracy": correct / len(df), "seconds": round(dt, 1),
                                   "truncated": sum(r["tokens"] >= 2048 for r in rows)}, rows)


if __name__ == "__main__":
    main()
