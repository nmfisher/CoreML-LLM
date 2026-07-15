#!/usr/bin/env python3
"""Bisection helper: build truncated MergedChunk23 partials at shrinking
shared_range to find the ANE size cliff (within the real merged structure).

own_range is held at the real L8-14 block (the merge's own-KV part); only
shared_range is shrunk, so every partial keeps the SAME merged topology that
fails at full size (17 layers) — isolating SIZE, not structure.

Reuses build_gemma4_3way's convert/save helpers verbatim (fp16, ios18,
skip_model_load=True, CPU_AND_NE) so the artifacts are directly comparable to
out_vfix/chunk2_3way.mlpackage.
"""
from __future__ import annotations
import argparse, os, sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from models.gemma4 import Gemma4Model
from models.gemma4_swa_merged import MergedChunk23
from build_gemma4_3way import _resolve_hf_dir, _convert_and_palettize, _save, fp16


def build_partial(base, ctx, out_pkg, *, own, shared, label):
    cfg = base.config
    hidden, pld, nlayers = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
    W, hd_s, hd_f = cfg.sliding_window, cfg.head_dim, cfg.global_head_dim
    nkv, max_hd = cfg.num_key_value_heads, cfg.global_head_dim

    mc = MergedChunk23(base, own_range=own, shared_range=shared).eval()
    ns, nf = mc.num_sliding, mc.num_full
    n_layers = (mc.END_C2 - mc.START_C2) + (mc.END_C3 - mc.START_C3)
    print(f"\n=== {label} (own L{own[0]}-{own[1]-1} + shared L{shared[0]}-{shared[1]-1}, "
          f"{n_layers} layers, ns={ns} nf={nf}) ===")

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=fp16),
    ]
    outputs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
               "K_full_out", "V_full_out",
               "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    m = _convert_and_palettize(mc, sample, inputs, outputs, label=label, quantize=False)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--shared-ends", default="20,17",
                    help="comma list of shared_range END (start fixed at 15). "
                         "own_range fixed at (8,15). E.g. 20 -> shared L15-19 (12 layers).")
    ap.add_argument("--own-start", type=int, default=8)
    ap.add_argument("--own-end", type=int, default=15)
    ap.add_argument("--shared-start", type=int, default=15)
    args = ap.parse_args()

    args.output = args.output or os.path.join(ROOT, "..", "output", args.model, "partials")
    os.makedirs(args.output, exist_ok=True)
    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx); base.eval()

    own = (args.own_start, args.own_end)
    for end in [int(e) for e in args.shared_ends.split(",") if e]:
        shared = (args.shared_start, end)
        n = (own[1]-own[0]) + (shared[1]-shared[0])
        label = f"chunk2_L{own[0]}-{own[1]-1}_L{shared[0]}-{shared[1]-1}_{n}lyr"
        out_pkg = os.path.join(args.output, f"{label}.mlpackage")
        build_partial(base, args.ctx, out_pkg, own=own, shared=shared, label=label)
    print("\nDONE")


if __name__ == "__main__":
    main()
