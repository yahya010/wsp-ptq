"""Build the BF16 copy that the benchmarks run on: build.py <out dir> [edit.pt ...]
= the start checkpoint with each edit's matrices replacing its own (w3.overlay), dequantized (w3.to_bf16)."""
import sys

import torch

import w3

torch.set_grad_enabled(False)
out, edits = sys.argv[1], sys.argv[2:]
base = w3.load()
st = w3.overlay(base, *edits)
print("changed:", w3.changed(base, st), flush=True)
print("wrote", w3.to_bf16(st, out))
