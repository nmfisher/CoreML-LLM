#!/usr/bin/env python3
"""Half-width chunk1 probe: rebuild chunk1 (L0-7) at HALF the residual width
to measure how ANE latency scales with per-layer compute.

Random weights (no pretrained load) — this is a pure compute-scaling probe;
weight VALUES do not change ANE dispatch/timing, only shapes do.

Halved (all proportionally, so every matmul has BOTH dims halved -> ~4x less
FLOPs, ~4x fewer weight params):
  hidden_size                  1536 -> 768
  head_dim                     256  -> 128
  global_head_dim              512  -> 256
  intermediate_size            6144 -> 3072
  hidden_size_per_layer_input  256  -> 128
  (num_attention_heads=8, num_key_value_heads=1 unchanged)

Compare chunk-probe `ne` latency vs the baseline fp16 chunk1 (full 1536 width,
592 MB, ~19 ms). If latency tracks compute, expect ~19/4 ~= 5 ms. If it barely
moves, per-op dispatch overhead dominates (consistent with the compute-bound
but dispatch-heavy picture in FINDINGS §14/§15).
"""
from __future__ import annotations
import argparse, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch

import coremltools as ct
from models.gemma4 import Gemma4Model, Gemma4Config
from models.gemma4_swa_chunks import SWAChunk1, compute_chunk_boundaries
from build_gemma4_3way import _convert_and_palettize, _save

fp16 = np.float16


def build_halfwidth_chunk1(ctx: int, out_pkg: str):
    cfg = Gemma4Config(
        hidden_size=768,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=128,
        global_head_dim=256,
        intermediate_size=3072,
        vocab_size=262144,
        sliding_window=512,
        rms_norm_eps=1e-6,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=128,
        context_length=ctx,
    )
    # Random init — no checkpoint load. torch.manual_seed not needed (values
    # are irrelevant to timing).
    base = Gemma4Model(cfg).eval()

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
    print(f"\n=== halfwidth chunk1 (L{c1_start}-{c1_end-1}), hidden={hidden} hd_s={hd_s} hd_f={hd_f} inter={cfg.intermediate_size} ===")

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
    m = _convert_and_palettize(swa1, s1, in1, out1, label="halfwidth_chunk1", quantize=False)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    args.output = args.output or os.path.join(ROOT, "..", "output", "gemma4-e2b", "chunks_halfwidth")
    os.makedirs(args.output, exist_ok=True)
    out_pkg = os.path.join(args.output, "chunk1.mlpackage")
    build_halfwidth_chunk1(args.ctx, out_pkg)
    print("\nDONE")


if __name__ == "__main__":
    main()
