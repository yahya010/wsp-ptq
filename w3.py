"""The W3 3-bit format, and the only code that reads or writes checkpoints. Do not modify.

A quantized matrix W (torch layout: [out_features, in_features], zero-padded to a multiple of 128 both ways) is stored as
  codes  uint8 [N, K]         one of 8 codebook levels per weight
  scale  float8_e4m3fn [N, K/32]  one scale per 32 consecutive weights of a row (a "group", along the input dimension)
  row    float32 [N]           one scale per row
  levels the 8-level codebook (None = the default LEVELS below)
  n, k   the true shape (the padding is cropped when the weight is used)
and dequantizes to  W[i, j] = levels[codes[i, j]] * scale[i, j // 32] * row[i].

In memory a checkpoint ("state") is a dict {matrix name: that dict}. Matrix names are module names, e.g.
"backbone.layers.0.mixer.in_proj", "lm_head", "backbone.layers.1.mixer.experts.7.up_proj". Every tensor that is not
quantized (embeddings, norms, router, shared experts, ...) comes from the BF16 reference unchanged.
"""
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LEVELS = torch.tensor([-3.5, -2.25, -1.25, -0.40625, 0.40625, 1.25, 2.25, 3.5])
ROOT = Path(os.environ.get("WSP_ROOT", Path(__file__).resolve().parent / "work"))
REF = ROOT / "ref"          # BF16 dequantization of the NVFP4 reference (prepare.py)
START = ROOT / "start"      # the 3-bit checkpoint to repair, in this format (prepare.py)


def pad128(x):
    return (x + 127) // 128 * 128


def levels(m):
    return LEVELS if m.get("levels") is None else m["levels"]


def deq(m):
    """Dequantized weight [n, k] (float32, on the device of m['codes'])."""
    lv = levels(m).to(m["codes"].device).float()
    W = lv[m["codes"].long()] * (m["scale"].float() * m["row"][:, None]).repeat_interleave(32, dim=1)
    return W[: m["n"], : m["k"]]


def clone(m):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in m.items()}


def to(m, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in m.items()}


def reference(name):
    """The reference (NVFP4, dequantized to BF16) weight of one matrix, float32 [n, k]."""
    global _REF_IDX
    if "_REF_IDX" not in globals():
        _REF_IDX = json.load(open(REF / "model.safetensors.index.json"))["weight_map"]
    with safe_open(REF / _REF_IDX[name + ".weight"], "pt") as f:
        return f.get_tensor(name + ".weight").float()


# ---------------------------------------------------------------- on-disk format (same as the published checkpoints)
def _unpack3(b, numel):
    b = b.reshape(-1, 3).to(torch.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    return torch.stack([(v >> (3 * i)) & 7 for i in range(8)], 1).reshape(-1)[:numel].to(torch.uint8)


def _pack3(codes):
    c = codes.reshape(-1, 8).to(torch.int32)
    v = sum(c[:, i] << (3 * i) for i in range(8))
    return torch.stack([(v >> s) & 0xFF for s in (0, 8, 16)], 1).to(torch.uint8).reshape(-1)


def load(d=START, names=None):
    """Load a checkpoint directory (w3-*.safetensors + w3.safetensors.index.json) as a state; `names` loads a subset."""
    d = Path(d)
    wmap = json.load(open(d / "w3.safetensors.index.json"))["weight_map"]
    all_names = sorted({k[: -len(".codes3")] for k in wmap if k.endswith(".codes3")})
    files, st = {}, {}

    def get(k):
        if wmap[k] not in files:
            files[wmap[k]] = safe_open(d / wmap[k], "pt")
        return files[wmap[k]].get_tensor(k)

    for n in (all_names if names is None else names):
        rows, cols, N, K = get(f"{n}.shape").tolist()
        lv = get(f"{n}.levels").float()
        st[n] = {"codes": _unpack3(get(f"{n}.codes3"), N * K).reshape(N, K), "scale": get(f"{n}.scale"),
                 "row": get(f"{n}.row").float(), "levels": None if torch.equal(lv, LEVELS) else lv, "n": rows, "k": cols}
    return st


def save(st, out):
    """Write a full state in the on-disk format (codes packed 8 per 3 bytes; unquantized tensors copied from REF)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    shard, size, n_shard, wmap = {}, 0, 0, {}

    def flush():
        nonlocal shard, size, n_shard
        if shard:
            n_shard += 1; fn = f"w3-{n_shard:05d}.safetensors"
            save_file(shard, out / fn, metadata={"format": "w3"}); wmap.update({k: fn for k in shard})
        shard, size = {}, 0

    def add(k, t):
        nonlocal size
        shard[k] = t.contiguous().clone(); size += t.numel() * t.element_size()
        if size > 5e9:
            flush()

    for n, m in st.items():
        add(f"{n}.codes3", _pack3(m["codes"])); add(f"{n}.scale", m["scale"]); add(f"{n}.row", m["row"].float())
        add(f"{n}.levels", levels(m).float())
        add(f"{n}.shape", torch.tensor([m["n"], m["k"], *m["codes"].shape], dtype=torch.int64))
    idx = json.load(open(REF / "model.safetensors.index.json"))["weight_map"]
    for f in sorted(set(idx.values())):
        with safe_open(REF / f, "pt") as s:
            for k in s.keys():
                if not (k.endswith(".weight") and k[:-7] in st):
                    add(k, s.get_tensor(k))
    flush()
    json.dump({"metadata": {"format": "w3", "quantized_matrices": len(st)}, "weight_map": wmap},
              open(out / "w3.safetensors.index.json", "w"), indent=1)
    _copy_aux(out)
    return out


def to_bf16(st, out, dev="cuda" if torch.cuda.is_available() else "cpu"):
    """BF16 Hugging Face copy of REF with every quantized matrix replaced by its dequantized value; vLLM and
    transformers load it directly. This is what every benchmark runs on."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(REF / "model.safetensors.index.json"))["weight_map"]
    for f in sorted(set(idx.values())):
        with safe_open(REF / f, "pt") as s:
            t = {}
            for k in s.keys():
                n = k[:-7] if k.endswith(".weight") else None
                if n in st:
                    t[k] = deq(to(st[n], dev)).to(torch.bfloat16).cpu()
                else:
                    t[k] = s.get_tensor(k)
        save_file(t, out / f, metadata={"format": "pt"})
    for fn in os.listdir(REF):
        if not fn.endswith(".safetensors"):
            shutil.copy(REF / fn, out / fn)
    return out


def _copy_aux(out):
    for fn in os.listdir(REF):
        if not fn.endswith(".safetensors") and fn != "model.safetensors.index.json":
            shutil.copy(REF / fn, out / fn)


# ---------------------------------------------------------------- edits
def overlay(base, *edits):
    """A new state: `base` with the matrices of each edit (a {name: matrix} dict, or a path to one saved with
    torch.save) replacing its own. Edits are small; save them, not whole states."""
    st = dict(base)
    for e in edits:
        st.update(torch.load(e, map_location="cpu") if isinstance(e, (str, Path)) else e)
    return st


def changed(base, st):
    """How much `st` differs from `base`: codes and group scales changed, and the share of all quantized weights."""
    codes = scales = 0
    for n, m in st.items():
        b = base[n]
        if m is b:
            continue
        codes += int((m["codes"] != b["codes"]).sum())
        scales += int((m["scale"].view(torch.uint8) != b["scale"].view(torch.uint8)).sum())
    total = sum(m["n"] * m["k"] for m in base.values())
    return {"codes": codes, "group_scales": scales, "share_of_weights": codes / total}
