#!/usr/bin/env python3
"""Tied-weight chunk1 probe (L0-7): replace every shape-compatible layer
weight in chunk1 with LAYER 0's weights, to test whether CoreML stores the
shared weights once or N times.

This is an ARCHITECTURE probe, not a correctness build. Tying L1-7 to L0 makes
the model emit garbage — that is expected and irrelevant. The point is to
measure, against the baseline fp16 chunk1.mlmodelc:

  1. Does the compiled .mlmodelc SHRINK when 7 layers share one layer's
     weights? (i.e. does CoreML / coremltools dedup shared parameter storage?)
  2. Does the chunk still run on the ANE, and at what latency vs baseline?

WHY ONLY SHAPE-COMPATIBLE PARAMS ARE TIED
-----------------------------------------
chunk1 = L0-7. Layer types follow (i+1)%5==0 -> full_attention, so L4 is a
full-attention layer (head_dim=512) while L0,1,2,3,5,6,7 are sliding
(head_dim=256). L4's q_proj/k_proj/v_proj/q_norm/k_norm/o_proj weights are a
different shape from L0's and CANNOT be assigned. The builder ties every
parameter whose name AND shape match L0's, and SKIPS (leaves unique) only the
mismatched attention projections on L4. L4's MLP / layernorms / per-layer
modules DO match and ARE tied.

The tying shares underlying storage (target.data = src.data), the strongest
form of PyTorch-level weight reuse — so if coremltools recognized shared
storage it would show up here.
"""
from __future__ import annotations
import argparse, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch

import coremltools as ct
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, compute_chunk_boundaries
from build_gemma4_3way import _resolve_hf_dir, _convert_and_palettize, _save

fp16 = np.float16


def tie_chunk1_to_layer0(model, start: int, end: int) -> dict:
    """Tie model.layers[start+1 .. end-1] params to model.layers[start].

    Returns a {layer_idx: {"tied": int, "skipped": [(name, src_shape, dst_shape)]}} report.
    """
    report = {}
    src_layer = model.layers[start]
    src_params = dict(src_layer.named_parameters())
    for idx in range(start + 1, end):
        dst = model.layers[idx]
        dst_params = dict(dst.named_parameters())
        tied = 0
        skipped = []
        for name, src_p in src_params.items():
            if name not in dst_params:
                continue
            dst_p = dst_params[name]
            if tuple(dst_p.shape) == tuple(src_p.shape):
                dst_p.data = src_p.data  # share storage
                tied += 1
            else:
                skipped.append((name, tuple(src_p.shape), tuple(dst_p.shape)))
        report[idx] = {"tied": tied, "skipped": skipped}
    return report


def build_tied_chunk1(base, ctx: int, out_pkg: str):
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    c1_start, c1_end = boundaries[0]
    print(f"\n=== tied chunk1 (L{c1_start}-{c1_end-1}), tying L{c1_start+1}-{c1_end-1} -> L{c1_start} ===")

    report = tie_chunk1_to_layer0(base, c1_start, c1_end)
    for idx, r in report.items():
        is_full = cfg.is_full_attention(idx)
        skip_str = "".join(f"\n      skip {n}: src{s} != dst{d}" for n, s, d in r["skipped"])
        print(f"    L{idx} ({'full' if is_full else 'sliding'}): tied {r['tied']} params{skip_str}")

    swa1 = SWAChunk1(base, c1_start, c1_end).eval()
    ns1, nf1 = swa1.num_sliding, swa1.num_full
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, ctx, max_hd, dtype=torch.float16),
    )
    in1 = [
        ct.TensorType(name="hidden_states",      shape=s1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s1[12].shape, dtype=fp16),
    ]
    out1 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "per_layer_combined_out"]
    # fp16, NO palettization — to compare like-for-like against the baseline
    # fp16 chunk1.mlmodelc (bundle_real4ch).
    m = _convert_and_palettize(swa1, s1, in1, out1, label="tied_chunk1", quantize=False)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    args.output = args.output or os.path.join(ROOT, "..", "output", args.model, "chunks_tied")
    os.makedirs(args.output, exist_ok=True)
    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    out_pkg = os.path.join(args.output, "chunk1.mlpackage")
    build_tied_chunk1(base, args.ctx, out_pkg)
    print("\nDONE")


if __name__ == "__main__":
    main()
