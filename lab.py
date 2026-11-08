"""Validation lab: the reference model in transformers on one GPU, with any quantized matrices swapped in. Do not modify.

    lab = Lab()                       # loads the reference and its outputs on the validation sets (~2 min)
    lab.set_state(w3.load())          # swap in all 3-bit matrices (or set_state(st, names=[...]) for a subset)
    lab.full()                        # KL to the reference on chat math / code / prose, perplexity, MMLU-val
    lab.restore(["lm_head"])          # put matrices back to their reference value
    X = lab.inputs("backbone.layers.0.mixer.out_proj", "prose", n=4)   # activations entering a matrix

Everything here is validation data. The test sets are used once, at the end, by evals/test_panel.sh.
"""
import json
import os

import torch

import w3

DATA = w3.ROOT / "data"


class Lab:
    def __init__(self, dev="cuda", n_math=96, n_code=32, n_prose=48, win=1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = dev
        self.model = AutoModelForCausalLM.from_pretrained(w3.REF, dtype=torch.bfloat16, device_map=dev).eval()
        self.tok = AutoTokenizer.from_pretrained(w3.REF)
        self.bb = getattr(self.model, "backbone", None) or self.model.model
        self.head = self.model.lm_head
        self.linears, self.experts = {}, {}
        for i, layer in enumerate(self.bb.layers):
            mx = layer.mixer
            if "Mamba" in type(mx).__name__:
                for p in ("in_proj", "out_proj"):
                    self.linears[f"backbone.layers.{i}.mixer.{p}"] = getattr(mx, p)
            elif hasattr(mx, "q_proj"):
                for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    self.linears[f"backbone.layers.{i}.mixer.{p}"] = getattr(mx, p)
            elif hasattr(mx, "experts"):
                self.experts[i] = mx.experts
        self.linears["lm_head"] = self.head
        self.names = list(self.linears) + [f"backbone.layers.{i}.mixer.experts.{e}.{p}" for i, ex in self.experts.items()
                                           for e in range(ex.up_proj.shape[0]) for p in ("up_proj", "down_proj")]
        self.orig = {n: lin.weight.data.to("cpu", copy=True) for n, lin in self.linears.items()}
        self.orig_up = {i: ex.up_proj.data.to("cpu", copy=True) for i, ex in self.experts.items()}
        self.orig_down = {i: ex.down_proj.data.to("cpu", copy=True) for i, ex in self.experts.items()}
        self._sets(n_math, n_code, n_prose, win)

    # ---------------------------------------------------------------- weights
    def weight(self, name, orig=True):
        """Reference weight (orig=True) or the one currently in the model, float32 [out, in]."""
        if name in self.linears:
            return (self.orig[name] if orig else self.linears[name].weight.data).float()
        i, e, p = self._expert(name)
        src = (self.orig_up if p == "up_proj" else self.orig_down)[i] if orig else getattr(self.experts[i], p).data
        return src[e].float()

    def _expert(self, name):
        parts = name.split(".")
        return int(parts[2]), int(parts[5]), parts[6]

    def set(self, name, W):
        if name in self.linears:
            self.linears[name].weight.data.copy_(W.to(torch.bfloat16))
        else:
            i, e, p = self._expert(name)
            getattr(self.experts[i], p).data[e].copy_(W.to(torch.bfloat16))

    def set_state(self, st, names=None):
        for n in (names or st):
            self.set(n, w3.deq(w3.to(st[n], self.dev)))

    def restore(self, names=None):
        if names is None:
            for n, lin in self.linears.items():
                lin.weight.data.copy_(self.orig[n])
            for i, ex in self.experts.items():
                ex.up_proj.data.copy_(self.orig_up[i]); ex.down_proj.data.copy_(self.orig_down[i])
        else:
            for n in names:
                self.set(n, self.weight(n).to(self.dev))

    # ---------------------------------------------------------------- validation sets; teacher = the reference model
    def _sets(self, n_math, n_code, n_prose, win):
        rows = [r for r in json.load(open(DATA / "gsm8k_klgen_ref.json"))["rows"] if r["tokens"] < 2048][:n_math]
        self.seqs = [(r["prompt_ids"] + r["output_ids"])[:win] for r in rows]
        self.starts = [len(r["prompt_ids"]) for r in rows]          # chat math: KL over the reference's own answer
        g0 = len(self.seqs)
        for text in json.load(open(DATA / "code_val.json"))[:n_code]:
            self.seqs.append(self.tok.encode(text, add_special_tokens=False)[:win]); self.starts.append(32)
        g1 = len(self.seqs)
        ids = self.tok(open(DATA / "wikitext2_val.txt").read()).input_ids
        self.prose_all = [ids[i:i + win] for i in range(0, len(ids) - win + 1, win)]
        for w in self.prose_all[:n_prose]:
            self.seqs.append(w); self.starts.append(32)
        self.groups = {"math": range(0, g0), "code": range(g0, g1), "prose": range(g1, len(self.seqs))}
        self.restore()
        self.teacher = self._hidden(range(len(self.seqs)))
        self.letters = [self.tok.encode(" " + c, add_special_tokens=False)[0] for c in "ABCD"]
        self.load_mmlu(DATA / "mmlu_val.json")

    def load_mmlu(self, path):
        self.mmlu_items = json.load(open(path))
        self.mmlu_prompts = [self.tok.encode(
            f"The following are multiple choice questions (with answers) about {it['subject'].replace('_', ' ')}.\n\n"
            + it["q"].strip() + "\n" + "\n".join(f"{c}. {x}" for c, x in zip("ABCD", it["choices"])) + "\nAnswer:",
            add_special_tokens=False) for it in self.mmlu_items]
        self.restore(); self.mmlu_teacher = self._mmlu_hidden()   # resets every matrix to the reference

    @torch.no_grad()
    def _hidden(self, idx):
        out = []
        for j in idx:
            h = self.bb(input_ids=torch.tensor([self.seqs[j]], device=self.dev)).last_hidden_state[0]
            out.append(h[self.starts[j] - 1: len(self.seqs[j]) - 1].clone())
        return out

    @torch.no_grad()
    def _mmlu_hidden(self, batch=4):
        order = sorted(range(len(self.mmlu_prompts)), key=lambda i: len(self.mmlu_prompts[i]))
        out = [None] * len(order)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]; L = max(len(self.mmlu_prompts[i]) for i in idx)
            x = torch.zeros(len(idx), L, dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(self.mmlu_prompts[i])] = torch.tensor(self.mmlu_prompts[i])
            h = self.bb(input_ids=x.to(self.dev)).last_hidden_state
            for r, i in enumerate(idx):
                out[i] = h[r, len(self.mmlu_prompts[i]) - 1].clone()
        return torch.stack(out)

    # ---------------------------------------------------------------- metrics
    @torch.no_grad()
    def kl(self, group, per_token=False):
        """Mean KL(reference || current) per token on one validation group, and top-1 agreement. per_token=True also
        returns the per-token KL of each sequence (to see where in a sequence the damage is)."""
        idx = list(self.groups[group]); hs = self._hidden(idx)
        tot, agree, n, each = 0.0, 0, 0, []
        for j, h in zip(idx, hs):
            t, k_seq = self.teacher[j], []
            for c in range(0, len(h), 512):
                lt = torch.log_softmax(self.head(t[c:c + 512]).float(), -1)
                ls = torch.log_softmax(self.head(h[c:c + 512]).float(), -1)
                k = (lt.exp() * (lt - ls)).sum(-1); k_seq.append(k.cpu())
                tot += float(k.sum()); agree += int((lt.argmax(-1) == ls.argmax(-1)).sum())
            n += len(h); each.append(torch.cat(k_seq))
        return (tot / n, agree / n, each) if per_token else (tot / n, agree / n)

    @torch.no_grad()
    def ppl(self):
        nll, n = 0.0, 0
        for w in self.prose_all:
            x = torch.tensor([w], device=self.dev); h = self.bb(input_ids=x).last_hidden_state[0]
            ls = torch.log_softmax(self.head(h[:-1]).float(), -1)
            nll -= float(ls.gather(1, x[0, 1:, None]).sum()); n += len(w) - 1
        return float(torch.exp(torch.tensor(nll / n)))

    @torch.no_grad()
    def mmlu(self):
        h = self._mmlu_hidden()
        ls = torch.log_softmax(self.head(h).float(), -1); lt = torch.log_softmax(self.head(self.mmlu_teacher).float(), -1)
        acc = float((ls[:, self.letters].argmax(-1).cpu() == torch.tensor([it["answer"] for it in self.mmlu_items])).float().mean())
        return acc, float((lt.exp() * (lt - ls)).sum(-1).mean())

    def full(self):
        r = {}
        for g in self.groups:
            k, a = self.kl(g); r[g] = round(k, 5); r[g + "_agree"] = round(a, 4)
        r["ppl"] = round(self.ppl(), 3)
        a, k = self.mmlu(); r["mmlu"] = round(a, 4); r["mmlu_kl"] = round(k, 5)
        return r

    # ---------------------------------------------------------------- activations
    @torch.no_grad()
    def inputs(self, name, group="prose", n=4):
        """Activations entering a dense matrix (Mamba in/out_proj, attention q/k/v/o, lm_head) on the first n sequences
        of a validation group: a list of [tokens, in_features] float32 CPU tensors. Routed experts see only the tokens
        routed to them; hook the experts module yourself for those."""
        got = []
        hook = self.linears[name].register_forward_hook(lambda m, i, o: got.append(i[0][0].float().cpu()))
        try:
            for j in list(self.groups[group])[:n]:
                self.bb(input_ids=torch.tensor([self.seqs[j]], device=self.dev))
        finally:
            hook.remove()
        return got
