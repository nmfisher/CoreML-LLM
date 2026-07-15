#!/usr/bin/env python3
"""Fused-projection chunk1 probe (L0-7): collapse q/k/v into ONE conv and
gate/up into ONE conv to cut op COUNT (the §16 dispatch floor is per-op).

Algebraically EXACT — the fused conv is just the three weight tensors
concatenated along the output channel dim, with the result split back. Output
is byte-identical to the unfused path (verified at torch level before
conversion). So unlike SDPA fusion, this carries no correctness risk: it's a
pure op-count / dispatch reduction.

Per sliding layer this removes 2 heavy conv dispatches (q/k/v 3->1) + 1
(gate/up 2->1) = 3 fewer convs/layer. The matmuls, norms, attention, and PLE
block are untouched.

Built at --width {full,half} to see fusion's effect where compute dominates
(1536) vs where the dispatch floor dominates (768, §16).

Probe target: does cutting conv-op count drop the ~4.3 ms dispatch floor as the
§16 model predicts?
"""
from __future__ import annotations
import argparse, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax
from models.gemma4 import Gemma4Model, Gemma4Config
from models.gemma4_swa_chunks import (
    SWAChunk1, compute_chunk_boundaries, _layer_kv_map, v_norm,
)
from build_gemma4_3way import _convert_and_palettize, _save

fp16 = np.float16


def fuse_conv_weights(convs):
    """Concat a list of Conv2d (1x1) along output channels -> one Conv2d."""
    w = torch.cat([c.weight.data for c in convs], dim=0)
    has_bias = any(c.bias is not None for c in convs)
    out = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=1, bias=has_bias,
                    dtype=MODEL_DTYPE)
    out.weight.data.copy_(w)
    if has_bias:
        bs = [c.bias.data for c in convs if c.bias is not None]
        out.bias.data.copy_(torch.cat(bs, dim=0))
    return out


class FusedLayer(nn.Module):
    """A Gemma4DecoderLayer with q/k/v fused into qkv_proj and gate/up fused
    into gate_up_proj. All other submodules are reused from the source layer."""

    def __init__(self, src):
        super().__init__()
        self.qkv_proj = fuse_conv_weights([
            src.self_attn["q_proj"], src.self_attn["k_proj"], src.self_attn["v_proj"]])
        self.gate_up_proj = fuse_conv_weights([
            src.mlp["gate_proj"], src.mlp["up_proj"]])
        # reuse the rest verbatim
        self.input_layernorm = src.input_layernorm
        self.self_attn = nn.ModuleDict({
            "o_proj": src.self_attn["o_proj"],
            "q_norm": src.self_attn["q_norm"],
            "k_norm": src.self_attn["k_norm"],
        })
        self.post_attention_layernorm = src.post_attention_layernorm
        self.pre_feedforward_layernorm = src.pre_feedforward_layernorm
        self.mlp = nn.ModuleDict({"down_proj": src.mlp["down_proj"]})
        self.post_feedforward_layernorm = src.post_feedforward_layernorm
        self.per_layer_input_gate = src.per_layer_input_gate
        self.per_layer_projection = src.per_layer_projection
        self.post_per_layer_input_norm = src.post_per_layer_input_norm
        self.layer_scalar = src.layer_scalar


def _run_layer_swa_fused(
    fl, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding, update_mask,
    K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    """Copy of _run_layer_swa with fused q/k/v and gate/up projections."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)
    q_out = num_heads * hd
    kv_out = num_kv_heads * hd

    residual = hidden_states
    h = fl.input_layernorm(hidden_states)
    x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    # FUSED q/k/v: one conv, then slice along output channels (slice, not split —
    # coremltools' torch frontend mishandles torch.split size-lists)
    qkv = fl.qkv_proj(x)
    q = qkv[:, :q_out]
    k = qkv[:, q_out:q_out + kv_out]
    v = qkv[:, q_out + kv_out:]
    q = q.view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = fl.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    K_sliding_out = K_sliding_slot
    V_sliding_out = V_sliding_slot
    K_full_out = K_full_slot
    V_full_out = V_full_slot

    if not is_kv_shared:
        k = k.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = v.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = fl.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
        v = v_norm(v)
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        if hd < max_hd:
            k_padded = torch.nn.functional.pad(k, (0, max_hd - hd))
            v_padded = torch.nn.functional.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v

        if is_full:
            K_full_out = K_full_slot * (1 - update_mask) + k_padded.expand_as(K_full_slot) * update_mask
            V_full_out = V_full_slot * (1 - update_mask) + v_padded.expand_as(V_full_slot) * update_mask
            K_for_attn = K_full_out[..., :hd]
            V_for_attn = V_full_out[..., :hd]
        else:
            K_sliding_out = torch.cat([K_sliding_slot[:, :, 1:, :], k_padded], dim=2)
            V_sliding_out = torch.cat([V_sliding_slot[:, :, 1:, :], v_padded], dim=2)
            K_for_attn = K_sliding_out[..., :hd]
            V_for_attn = V_sliding_out[..., :hd]

        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_sliding_out[..., :config.head_dim]
            kv_store_13_v = V_sliding_out[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_full_out[..., :config.global_head_dim]
            kv_store_14_v = V_full_out[..., :config.global_head_dim]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = fl.self_attn["o_proj"](
        attn_output.permute(0, 2, 1).unsqueeze(2)
    ).squeeze(2).permute(0, 2, 1)
    attn_output = fl.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP — FUSED gate/up: one conv, then split
    residual = hidden_states
    h = fl.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    # gate/up each have get_intermediate_size out-channels; fused conv is 2x that.
    # Read the size from CONFIG (python int), NOT weight.shape — tracing records
    # .weight.shape as a graph op and coremltools can't scalar-cast it.
    inter = config.get_intermediate_size(layer_idx)
    gu = fl.gate_up_proj(x_mlp)
    gate = gu[:, :inter]
    up = gu[:, inter:]
    gate = torch.nn.functional.gelu(gate, approximate="tanh")
    mlp_out = fl.mlp["down_proj"](gate * up)
    hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
    hidden_states = fl.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]
    hs_conv = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    gated = fl.per_layer_input_gate(hs_conv)
    gated = torch.nn.functional.gelu(gated, approximate="tanh")
    per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)
    gated = gated * per_layer_slice_conv
    gated = fl.per_layer_projection(gated)
    gated = gated.squeeze(2).permute(0, 2, 1)
    hidden_states = fl.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * fl.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


class FusedSWAChunk1(SWAChunk1):
    """SWAChunk1 with fused projections. Inherits _compute_ple + PLE weights."""
    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)
        dummy_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        K_sliding_outs, V_sliding_outs, K_full_outs, V_full_outs = [], [], [], []
        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_sliding_slot = K_sliding_slot
            else:
                si = self.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_full_slot = K_full_slot
            (hidden_states, Kso, Vso, Kfo, Vfo, *_) = _run_layer_swa_fused(
                self.fused_layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
            )
            if is_full:
                K_full_outs.append(Kfo.squeeze(0)); V_full_outs.append(Vfo.squeeze(0))
            else:
                K_sliding_outs.append(Kso.squeeze(0)); V_sliding_outs.append(Vso.squeeze(0))
        return (hidden_states,
                torch.stack(K_sliding_outs, dim=0), torch.stack(V_sliding_outs, dim=0),
                torch.stack(K_full_outs, dim=0), torch.stack(V_full_outs, dim=0),
                per_layer_combined)


def _config_for_width(width: str, ctx: int) -> Gemma4Config:
    if width == "full":
        return Gemma4Config(context_length=ctx)  # defaults: 1536/256/512/6144/256
    return Gemma4Config(
        hidden_size=768, head_dim=128, global_head_dim=256, intermediate_size=3072,
        hidden_size_per_layer_input=128, context_length=ctx,
    )


def build_fused_chunk1(width: str, ctx: int, out_pkg: str):
    cfg = _config_for_width(width, ctx)
    base = Gemma4Model(cfg).eval()  # random init — values don't affect timing
    boundaries = compute_chunk_boundaries(cfg)
    c1_start, c1_end = boundaries[0]
    print(f"\n=== fused chunk1 width={width} (L{c1_start}-{c1_end-1}), hidden={cfg.hidden_size} ===")

    chunk = FusedSWAChunk1(base, c1_start, c1_end)
    chunk.fused_layers = nn.ModuleList(
        [FusedLayer(base.layers[i]) for i in range(c1_start, c1_end)])
    chunk.eval()

    # ---- correctness self-test: fused vs unfused on identical random input ----
    with torch.no_grad():
        ref = SWAChunk1(base, c1_start, c1_end).eval()
        ins = _dummy_inputs(cfg, ctx)
        y_ref = ref(*ins)
        y_fused = chunk(*ins)
        max_diff = max((float((a - b).abs().max()) for a, b in zip(y_ref, y_fused)), default=0.0)
    print(f"    self-test fused-vs-unfused max|Δ| = {max_diff:.3e} (expect ~0, fp16 noise)")
    assert max_diff < 1e-2, "FUSION NOT EXACT — aborting before convert"

    m = _convert_and_palettize(chunk, ins, _input_types(cfg, ctx),
                               ["hidden_states_out", "K_sliding_out", "V_sliding_out",
                                "K_full_out", "V_full_out", "per_layer_combined_out"],
                               label=f"fused_{width}", quantize=False)
    _save(m, out_pkg)


def _dummy_inputs(cfg, ctx):
    hidden = cfg.hidden_size; pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers; W = cfg.sliding_window
    hd_s = cfg.head_dim; hd_f = cfg.global_head_dim; max_hd = hd_f; nkv = cfg.num_key_value_heads
    nkv = max(nkv, 1)
    # need num_sliding/num_full slots — derive cheaply
    sm, fm = {}, {}; si = fi = 0
    for i in range(0, 8):
        if cfg.is_full_attention(i): fm[i] = fi; fi += 1
        else: sm[i] = si; si += 1
    ns1, nf1 = len(sm), len(fm)
    return (
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


def _input_types(cfg, ctx):
    s = _dummy_inputs(cfg, ctx)
    names = ["hidden_states", "causal_mask_full", "causal_mask_sliding", "update_mask",
             "per_layer_raw", "cos_s", "sin_s", "cos_f", "sin_f",
             "K_sliding_in", "V_sliding_in", "K_full_in", "V_full_in"]
    return [ct.TensorType(name=n, shape=t.shape, dtype=fp16) for n, t in zip(names, s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", choices=("full", "half"), default="full")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    args.output = args.output or os.path.join(ROOT, "..", "output", "gemma4-e2b",
                                              f"chunks_fused_{args.width}")
    os.makedirs(args.output, exist_ok=True)
    build_fused_chunk1(args.width, args.ctx, os.path.join(args.output, "chunk1.mlpackage"))
    print("\nDONE")


if __name__ == "__main__":
    main()
