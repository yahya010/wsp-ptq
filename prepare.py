"""One-time setup. Do not modify. Everything lands in work/ (or $WSP_ROOT).

    python prepare.py ref                     # NVFP4 reference -> BF16 copy, work/ref (~60 GB; a GPU makes it fast)
    python prepare.py start --checkpoint poor # the 3-bit checkpoint to repair -> work/start (~17 GB)
    python prepare.py data                    # validation + test sets -> work/data (needs work/ref and one GPU)

--checkpoint: "poor" (poorly calibrated, the larger gap) or "well" (well-calibrated GPTQ). Both are 3-bit W3
checkpoints of the same reference; see README.md for their benchmark numbers.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

import w3

REFERENCE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
STARTS = {"poor": "yemara/Nemotron-3.5-Lightning-30B-A3B-W3-PoorlyCalibrated",
          "well": "yemara/Nemotron-3.5-Lightning-30B-A3B-W3-WellCalibrated"}
DATA = w3.ROOT / "data"


def ref(dev):
    """Dequantize the NVFP4 release to BF16 with the same tensor names. NVFP4 weights: uint8 [N, K/2], two E2M1 codes
    per byte (low nibble = even column) x weight_scale (E4M3 per 16) x weight_scale_2 (fp32). FP8 weights (Mamba):
    E4M3 x weight_scale (fp32). Everything else is copied."""
    src = Path(snapshot_download(REFERENCE)); dst = w3.REF; dst.mkdir(parents=True, exist_ok=True)
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)

    def nvfp4(w, s, s2):
        v = torch.stack([e2m1[(w & 0xF).long()], e2m1[(w >> 4).long()]], -1).reshape(w.shape[0], w.shape[1] * 2)
        return (v.reshape(v.shape[0], -1, 16) * s.float()[:, :, None]).reshape(v.shape) * s2.float()

    idx = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    newmap = {}
    for f in sorted(set(idx.values())):
        out = {}
        with safe_open(src / f, "pt", device=dev) as s:
            keys = set(s.keys())
            for k in keys:
                if k.endswith((".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale")):
                    continue
                t = s.get_tensor(k); base = k[:-7] if k.endswith(".weight") else None
                if base and f"{base}.weight_scale_2" in keys:
                    t = nvfp4(t, s.get_tensor(f"{base}.weight_scale"), s.get_tensor(f"{base}.weight_scale_2"))
                elif base and f"{base}.weight_scale" in keys:
                    t = t.float() * s.get_tensor(f"{base}.weight_scale").float()
                out[k] = t.to(torch.bfloat16).cpu() if t.is_floating_point() else t.cpu(); newmap[k] = f
        save_file(out, dst / f, metadata={"format": "pt"}); print(f, len(out), flush=True)
    json.dump({"metadata": {}, "weight_map": newmap}, open(dst / "model.safetensors.index.json", "w"))
    cfg = json.load(open(src / "config.json")); cfg.pop("quantization_config", None); cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    for fn in os.listdir(src):
        if fn.endswith((".json", ".jinja", ".py")) and fn not in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
            shutil.copy(src / fn, dst / fn)


def start(which):
    snapshot_download(STARTS[which], local_dir=w3.START)
    json.dump({"checkpoint": which, "repo": STARTS[which]}, open(w3.START / "wsp_start.json", "w"))


def hub(repo, pattern):
    d = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])
    return sorted(glob.glob(f"{d}/{pattern}"))


def data():
    data_files(); teacher()


def data_files():
    import pandas as pd
    DATA.mkdir(parents=True, exist_ok=True)
    # prose: WikiText-2 (raw) validation and test, joined as in the usual perplexity setup
    for split, name in (("validation", "val"), ("test", "test")):
        df = pd.concat(pd.read_parquet(p) for p in hub("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-*.parquet"))
        open(DATA / f"wikitext2_{name}.txt", "w").write("\n\n".join(df.text))
    # MMLU: validation + dev (1,816) for selection, test (14,042) once
    for name, splits in (("val", ("validation", "dev")), ("test", ("test",))):
        df = pd.concat(pd.read_parquet(p) for s in splits for p in hub("cais/mmlu", f"all/{s}-*.parquet"))
        json.dump([{"q": r.question, "choices": list(r.choices), "answer": int(r.answer), "subject": r.subject} for r in df.itertuples()],
                  open(DATA / f"mmlu_{name}.json", "w"))
    # code: 32 fixed Python files from this interpreter's standard library (stored as text, so the set never changes)
    lib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in lib.glob("*.py") if 8_000 < p.stat().st_size < 60_000)
    json.dump([p.read_text() for p in random.Random(0).sample(files, 32)], open(DATA / "code_val.json", "w"))
    json.dump({"python": sys.version.split()[0]}, open(DATA / "meta.json", "w"))


def teacher():
    # chat math: the reference's own greedy answers to 256 GSM8K train questions (not the dev1k ones)
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "evals/gsm8k.py"), str(w3.REF), "ref", "klgen"], check=True, env={**os.environ, "TP": "1"})
    shutil.copy(w3.ROOT / "results" / "gsm8k-klgen-ref.json", DATA / "gsm8k_klgen_ref.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ref", "start", "data"])
    ap.add_argument("--checkpoint", choices=list(STARTS), default="poor")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"ref": lambda: ref(a.device), "start": lambda: start(a.checkpoint), "data": data}[a.step]()
