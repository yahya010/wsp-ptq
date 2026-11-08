"""LiveCodeBench v6 (code_generation_lite test6.jsonl, contest_date >= 2025-02-01: 131 problems), offline in vLLM with the
official lcb_runner prompt, code extraction and grader (every public + hidden test, 6 s each): lcb.py <model dir> <tag>
Needs a LiveCodeBench checkout: git clone https://github.com/LiveCodeBench/LiveCodeBench && git -C LiveCodeBench checkout
28fef95, then LCB_DIR=<that dir>. Thinking on, the model's default sampling, 3 samples per problem, 32K-token budget;
pass@1 = mean over samples. Run once, at the end."""
import json
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

MODEL = os.path.abspath(sys.argv[1])   # lcb_runner expects to run from its checkout
sys.path.insert(0, os.environ["LCB_DIR"])
os.chdir(os.environ["LCB_DIR"])
sys.modules.setdefault("anthropic", types.SimpleNamespace(HUMAN_PROMPT="\n\nHuman:", AI_PROMPT="\n\nAssistant:"))
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness  # noqa: E402
from lcb_runner.lm_styles import LMStyle  # noqa: E402
from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer  # noqa: E402
from lcb_runner.utils.extraction_utils import extract_code  # noqa: E402

from common import hf_file, llm, write  # noqa: E402


def problems():
    rows = [json.loads(line) for line in open(hf_file("livecodebench/code_generation_lite", "test6.jsonl"), "rb").read().splitlines() if line.strip()]
    return [CodeGenerationProblem(**r) for r in rows if r["contest_date"] >= "2025-02-01"]


def grade(args):
    p, code = args
    try:
        res, _ = check_correctness(p.get_evaluation_sample(), code, timeout=6, debug=False)
    except Exception:
        res = [-2]
    return bool(res) and all(r is True for r in res)


def main():
    model, tag = MODEL, sys.argv[2]
    n, max_tokens = 3, 32768
    probs = problems()
    m = llm(model, max_tokens + 4096, util=0.95)
    tok = m.get_tokenizer()
    sp = m.get_default_sampling_params(); sp.n, sp.max_tokens, sp.seed = n, max_tokens, 0
    prompts = [tok.apply_chat_template([{"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                                        {"role": "user", "content": get_generic_question_template_answer(p)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in probs]
    t = time.time(); outs = m.generate(prompts, sp); gen_s = time.time() - t
    jobs, rows = [], []
    for p, o in zip(probs, outs):
        for c in o.outputs:
            jobs.append((p, extract_code(c.text.split("</think>")[-1], LMStyle.OpenAIChat)))
            rows.append({"id": p.question_id, "difficulty": p.difficulty.value, "tokens": len(c.token_ids), "capped": c.finish_reason == "length"})
    with ProcessPoolExecutor(32) as pool:
        for r, ok in zip(rows, pool.map(grade, jobs, chunksize=1)):
            r["correct"] = ok
    write(f"lcb-{tag}", {"tag": tag, "problems": len(probs), "samples": len(rows), "pass@1": sum(r["correct"] for r in rows) / len(rows),
                         "capped": sum(r["capped"] for r in rows), "gen_seconds": round(gen_s, 1)}, rows)


if __name__ == "__main__":
    main()
