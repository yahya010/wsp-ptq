"""GPQA Diamond (198 questions; gated dataset Idavidrein/gpqa, accept its terms on the Hub first): gpqa.py <model dir> <tag>
Options shuffled per question with a seed from its text (same order for every checkpoint). Thinking on, the model's
default sampling, 4 samples per question, 32K-token budget; accuracy = mean over samples. Run once, at the end."""
import csv
import random
import re
import sys
import time
import zlib

from common import hf_file, llm, write

LETTERS = "ABCD"
ANSWER = re.compile(r"(?:Answer|answer is)\s*[:：]?\s*[<(\[*{]*\s*\(?\**([A-J])\b")


def items():
    out = []
    for r in csv.DictReader(open(hf_file("Idavidrein/gpqa", "gpqa_diamond.csv"), encoding="utf-8")):
        q = r["Question"].strip()
        options = [r["Correct Answer"].strip()] + [r[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
        order = list(range(4)); random.Random(zlib.crc32(q.encode())).shuffle(order)
        out.append({"id": r["Record ID"], "question": q, "options": [options[i] for i in order], "gold": LETTERS[order.index(0)]})
    return out


def main():
    model, tag = sys.argv[1:3]
    n, max_tokens = 4, 32768
    its = items()
    m = llm(model, max_tokens + 2048, util=0.95, max_seqs=384)
    tok = m.get_tokenizer()
    sp = m.get_default_sampling_params(); sp.n, sp.max_tokens, sp.seed = n, max_tokens, 0
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{it['question']}\n\n" + "\n".join(
        f"({LETTERS[i]}) {o}" for i, o in enumerate(it["options"])) + "\n\nThink step by step, then finish with a line containing only: Answer: <letter>"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in its]
    t = time.time(); outs = m.generate(prompts, sp)
    rows = []
    for it, o in zip(its, outs):
        for c in o.outputs:
            found = ANSWER.findall(c.text.split("</think>")[-1]); pred = found[-1] if found else None
            rows.append({"id": it["id"], "gold": it["gold"], "pred": pred, "ok": pred == it["gold"], "tokens": len(c.token_ids),
                         "capped": c.finish_reason == "length"})
    write(f"gpqa-{tag}", {"tag": tag, "questions": len(its), "samples": len(rows), "accuracy": sum(r["ok"] for r in rows) / len(rows),
                          "capped": sum(r["capped"] for r in rows), "seconds": round(time.time() - t, 1)}, rows)


if __name__ == "__main__":
    main()
