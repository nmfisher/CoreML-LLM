"""Qwen3-VL 8B chunk_0 with DeepStack vision injection — stateful.

Fork of build_qwen3_vl_2b_stateful_chunk0_vision.py, retargeted to the 8B
(and reused by the thin 4B fork). Produces `chunk_0_vision.mlpackage` that
drops in alongside the plain stateful `chunk_0.mlpackage` with the SAME
kv_cache_0 state shape (so the Swift MLState created from chunk_0 is
interchangeable). The generator routes chunk[0] through this variant when
an image is present; DeepStack features are added at text layers 0/1/2.

layers_per_chunk and the output subdir derive from the model config /
MODEL_ID (6 layers/chunk for 4B/8B). INT4 grouped gs=64 matches the body.

Inputs (vs plain stateful chunk_0): + ds_0,ds_1,ds_2 (1,1,hidden) fp16
DeepStack features, + visual_active (1,) fp32 gate (1.0 on image-pad steps).

Usage:
  python build_qwen3_vl_8b_stateful_chunk0_vision.py \\
      --out-dir /tmp/qwen3vl8b_stateful --num-chunks 6 --nbits 4 --group-size 64
"""
from pathlib import Path
import argparse
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE
import build_qwen3_vl_8b_stateful_chunks as S
from build_qwen3_vl_8b_stateful_chunks import ANEStatefulDecoderLayer

DEEPSTACK_LAYER_COUNT = 3


class DeepStackStatefulChunk0(nn.Module):
    """Stateful chunk_0 with DeepStack injection at layers 0/1/2."""
    def __init__(self, cfg, hf_layers, max_seq, layers_per_chunk):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.layers_per_chunk = layers_per_chunk

        self.layers = nn.ModuleList([
            ANEStatefulDecoderLayer(cfg, hf_layers[li], max_seq, li)
            for li in range(layers_per_chunk)
        ])
        # Same unified KV cache shape as plain stateful chunk_0 → MLState
        # handles are interchangeable between chunk_0 and chunk_0_vision.
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(2 * layers_per_chunk, cfg.num_key_value_heads,
                        max_seq, cfg.head_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos,
                ds_0, ds_1, ds_2, visual_active):
        h = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        deepstack = [ds_0, ds_1, ds_2]
        gate = visual_active.to(MODEL_DTYPE).view(1, 1, 1, 1)
        for li, layer in enumerate(self.layers):
            h = layer(h, cos, sin, causal_mask, current_pos, self.kv_cache_0)
            if li < DEEPSTACK_LAYER_COUNT:
                ds = deepstack[li]
                ds_conv = ds.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
                h = h + gate * ds_conv
        return h.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)


def _audit_ane(out_path: Path) -> float:
    reloaded = ct.models.MLModel(str(out_path),
                                 compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    dev = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = ("const" if (a is None and op.operator_name == "const")
                 else (a.preferred_compute_device.__class__.__name__ if a else "unknown"))
            dev[d] += 1
    total = sum(dev.values())
    compute = total - dev.get("const", 0)
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    pct = 100 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)")
    return pct


def convert(chunk, cfg, max_seq, layers_per_chunk, out_path: Path):
    print(f"\n--- convert STATEFUL chunk_0_vision ({layers_per_chunk} layers) ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size

    example = (
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),       # hidden_in
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),      # cos
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),      # sin
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),    # causal_mask
        torch.zeros(1, dtype=torch.int32),                   # current_pos
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_0
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_1
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_2
        torch.zeros(1, dtype=torch.float32),                 # visual_active
    )
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, max_seq), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
        ct.TensorType(name="ds_0", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_1", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_2", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="visual_active", shape=(1,), dtype=np.float32),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    state_shape = (2 * layers_per_chunk, cfg.num_key_value_heads,
                   max_seq, cfg.head_dim)
    ct_states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
            name="kv_cache_0",
        )
    ]

    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*')
                   if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def palettize_pkg(fp16_pkg: Path, out_pkg: Path, nbits: int, group_size: int = 64):
    if nbits == 4:
        op_cfg = OpPalettizerConfig(mode="kmeans", nbits=4,
                                    granularity="per_grouped_channel",
                                    group_size=group_size)
        gran = f"per_grouped_channel(gs={group_size})"
    else:
        op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits,
                                    granularity="per_tensor")
        gran = "per_tensor"
    print(f"\n--- palettize INT{nbits} [{gran}]: {fp16_pkg.name} → {out_pkg.name} ---")
    m_in = ct.models.MLModel(str(fp16_pkg))
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    m_out = palettize_weights(m_in, opt_cfg)
    m_out.save(str(out_pkg))
    dst_mb = sum(f.stat().st_size for f in out_pkg.rglob('*')
                  if f.is_file()) / 1e6
    print(f"  saved int{nbits} {out_pkg.name} ({dst_mb:.0f} MB)")
    _audit_ane(out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--num-chunks", type=int, default=6,
                    help="must match the body build so chunk_0_vision has the "
                         "same layers (and KV state) as chunk_0")
    ap.add_argument("--nbits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    _size = "8b" if "8B" in S.MODEL_ID else ("4b" if "4B" in S.MODEL_ID else "2b")
    chunks_dir = out_root / f"qwen3_vl_{_size}_stateful_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {S.MODEL_ID} text backbone (fp32) for chunk_0_vision...")
    cfg = S.load_text_config()
    layers_per_chunk = cfg.num_hidden_layers // args.num_chunks
    print(f"  layers_per_chunk={layers_per_chunk} hidden={cfg.hidden_size}")
    text_model, _lm = S.load_text_backbone()
    chunk0 = DeepStackStatefulChunk0(cfg, text_model.layers, args.max_seq,
                                     layers_per_chunk).eval().to(MODEL_DTYPE)
    del text_model, _lm

    fp16_path = fp16_dir / "chunk_0_vision.mlpackage"
    final_path = chunks_dir / "chunk_0_vision.mlpackage"
    convert(chunk0, cfg, args.max_seq, layers_per_chunk, fp16_path)
    if final_path.exists():
        shutil.rmtree(final_path)
    if args.nbits == 0:
        shutil.move(str(fp16_path), str(final_path))
    else:
        palettize_pkg(fp16_path, final_path, args.nbits, args.group_size)
    if not args.keep_fp16:
        shutil.rmtree(fp16_path, ignore_errors=True)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)
    print(f"\n✓ shipping artifact: {final_path}")


if __name__ == "__main__":
    main()
