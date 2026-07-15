#!/usr/bin/env python3
"""Decode-only 4-chunk Gemma 4 E2B build (NON-merged Topology I).

    chunk1: L0-7   SWAChunk1   (own KV)        — 8 layers
    chunk2: L8-14  SWAChunk2   (own KV)        — 7 layers   [was merged into chunk2_3way]
    chunk3: L15-24 SWAChunk3   (KV-shared)     — 10 layers  [was merged into chunk2_3way]
    chunk4: L25-34 SWAChunk4   (KV-shared + head) — 10 layers + head

Motivation: the 3-chunk "Topology II" merge of L8-24 into one chunk2_3way
(17 layers, 1714 ops) is rejected by the M2 Pro ANE at runtime (error -1) even
though the planner scores it 100% ANE. A size bisection proved the cliff is
between 12 and 17 layers — the SAME merged topology runs on ANE at 9 and 12
layers. So the fix is topological: keep L8-24 as TWO chunks (7 + 10 layers),
each under the ceiling, instead of one 17-layer merge.

IO contracts for SWAChunk2/SWAChunk3 are copied verbatim from
build_verify_chunks.py's [decode_q1] sections (the canonical 4-chunk builder
ChunkedEngine was designed against). chunk1/chunk4 reuse build_gemma4_3way's
build_chunk1 / build_chunk3_head (SWAChunk1 / SWAChunk4) unchanged — same
modules, same layers, as the working 3-chunk bundle.

The runtime detects 4-chunk mode by chunk2.mlpackage present (NOT chunk2_3way);
ChunkedEngine.load's `!is3Chunk` branch names them chunk1/2/3/4.
"""
from __future__ import annotations
import argparse, os, sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk2, SWAChunk3, compute_chunk_boundaries
from build_gemma4_3way import (
    _resolve_hf_dir, _convert_and_palettize, _save, fp16,
    build_chunk1, build_chunk3_head,
)


def build_chunk2_4ch(base, ctx, out_pkg, *, quantize):
    cfg = base.config
    hidden, pld, nlayers = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
    W, hd_s, hd_f = cfg.sliding_window, cfg.head_dim, cfg.global_head_dim
    nkv, max_hd = cfg.num_key_value_heads, cfg.global_head_dim
    boundaries = compute_chunk_boundaries(cfg)
    c2_start, c2_end = boundaries[1]
    swa2 = SWAChunk2(base, c2_start, c2_end).eval()
    ns2, nf2 = swa2.num_sliding, swa2.num_full
    print(f"\n=== chunk2 (L{c2_start}-{c2_end-1}, {c2_end-c2_start} layers, ns={ns2} nf={nf2}) ===")
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, ctx, max_hd, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=s[12].shape, dtype=fp16),
    ]
    outputs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
               "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    m = _convert_and_palettize(swa2, s, inputs, outputs, label="chunk2", quantize=quantize)
    _save(m, out_pkg)


def build_chunk3_4ch(base, ctx, out_pkg, *, quantize):
    cfg = base.config
    hidden, pld, nlayers = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
    W, hd_s, hd_f = cfg.sliding_window, cfg.head_dim, cfg.global_head_dim
    nkv = cfg.num_key_value_heads
    boundaries = compute_chunk_boundaries(cfg)
    c3_start, c3_end = boundaries[2]
    swa3 = SWAChunk3(base, c3_start, c3_end).eval()
    print(f"\n=== chunk3 (L{c3_start}-{c3_end-1}, {c3_end-c3_start} layers, KV-shared) ===")
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[12].shape, dtype=fp16),
    ]
    m = _convert_and_palettize(swa3, s, inputs, ["hidden_states_out"],
                               label="chunk3", quantize=quantize)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--only", choices=("chunk1", "chunk2", "chunk3", "chunk4"),
                    default=None, help="Build only one chunk (disk-friendly)")
    args = ap.parse_args()

    args.output = args.output or os.path.join(ROOT, "..", "output", args.model, "chunks_4ch")
    os.makedirs(args.output, exist_ok=True)
    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx); base.eval()
    quantize = not args.no_quantize

    c1 = os.path.join(args.output, "chunk1.mlpackage")
    c2 = os.path.join(args.output, "chunk2.mlpackage")
    c3 = os.path.join(args.output, "chunk3.mlpackage")
    c4 = os.path.join(args.output, "chunk4.mlpackage")
    if args.only in (None, "chunk1"): build_chunk1(base, args.ctx, c1, quantize=quantize)
    if args.only in (None, "chunk2"): build_chunk2_4ch(base, args.ctx, c2, quantize=quantize)
    if args.only in (None, "chunk3"): build_chunk3_4ch(base, args.ctx, c3, quantize=quantize)
    if args.only in (None, "chunk4"): build_chunk3_head(base, args.ctx, c4, quantize=quantize)
    print("\nDONE")


if __name__ == "__main__":
    main()
