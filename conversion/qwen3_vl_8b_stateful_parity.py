"""Mac parity / speed bench for the STATEFUL Qwen3-VL 8B chunks.

Drives the MLState (slice_update) chunks produced by
`build_qwen3_vl_8b_stateful_chunks.py` — the shippable artifact the
Swift `Qwen3VL2BStatefulGenerator(cfg: .default8B)` loads. Per chunk
inputs match the Swift runtime: hidden_in, cos, sin, causal_mask
(1,1,1,max_seq), current_pos (int32) + a Core ML-managed kv_cache_0
state created once per generation.

This is the stateful analogue of `qwen3_vl_8b_parity_ane.py`; both share
the embed sidecar and should produce the same greedy text (the only
difference is KV storage: MLState slice_update vs torch.where I/O).

Usage:
  python qwen3_vl_8b_stateful_parity.py \\
      --chunks-dir /tmp/qwen3vl8b_stateful/qwen3_vl_8b_stateful_chunks
"""
from pathlib import Path
import argparse
import time

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from build_qwen3_vl_8b_stateful_chunks import (
    load_text_config, MODEL_ID,
    EMBED_BIN_NAME, HEAD_CHUNK_NAME, _body_boundaries, NUM_BODY_CHUNKS,
)


PROMPTS = [
    "What is the capital of France?",
    "こんにちは、元気ですか?",
    "If a train leaves at 9:00 going 60 km/h and another leaves at 9:30 going 90 km/h from 150 km away, when do they meet?",
]


def rope_cos_sin_for_position(cfg, position: int):
    head_dim = cfg.head_dim
    theta = cfg.rope_scaling["rope_theta"]
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    angles = position * freqs
    full = np.concatenate([angles, angles])
    cos = np.cos(full).astype(np.float16).reshape(1, 1, head_dim)
    sin = np.sin(full).astype(np.float16).reshape(1, 1, head_dim)
    return cos, sin


def causal_mask_for_position(position: int, max_seq: int):
    m = np.full((1, 1, 1, max_seq), -1e4, dtype=np.float16)
    m[0, 0, 0, :position + 1] = 0.0
    return m


def run_chunked_decode(embed_weight, body_mlms, head_mlm, cfg,
                        max_seq, input_ids_list, max_new, eos_tokens=None):
    if eos_tokens is None:
        eos = getattr(cfg, "eos_token_id", None)
        eos_tokens = {eos} if eos is not None else set()

    # One MLState per body chunk, created fresh for this generation.
    states = [m.make_state() for m in body_mlms]

    def _step(tok_id: int, pos: int) -> int:
        hidden = embed_weight[tok_id:tok_id + 1, :].reshape(
            1, 1, -1).astype(np.float16)
        cos_np, sin_np = rope_cos_sin_for_position(cfg, pos)
        mask_np = causal_mask_for_position(pos, max_seq)
        pos_np = np.array([pos], dtype=np.int32)
        for mlm, st in zip(body_mlms, states):
            out = mlm.predict({
                "hidden_in": hidden, "cos": cos_np, "sin": sin_np,
                "causal_mask": mask_np, "current_pos": pos_np,
            }, state=st)
            hidden = out["hidden"].astype(np.float16)
        head_out = head_mlm.predict({"hidden_in": hidden})
        return int(head_out["next_token"].flatten()[0])

    last_token = None
    for t, tok_id in enumerate(input_ids_list):
        last_token = _step(tok_id, t)

    generated = []
    next_token = last_token
    S_prompt = len(input_ids_list)
    for step in range(max_new):
        pos = S_prompt + step
        if pos >= max_seq:
            break
        generated.append(next_token)
        if next_token in eos_tokens:
            break
        next_token = _step(next_token, pos)
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    embed_bin_path = chunks_dir / EMBED_BIN_NAME
    body_paths = [chunks_dir / f"chunk_{i}.mlpackage"
                  for i in range(args.num_chunks)]
    head_path = chunks_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    for p in [embed_bin_path, head_path, *body_paths]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print("loading config + tokenizer...")
    cfg = load_text_config()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"loading embed ({embed_bin_path.stat().st_size/1e6:.0f} MB) + "
          f"{args.num_chunks} stateful body chunks + head (Mac CPU+ANE)...")
    t0 = time.time()
    embed_weight = np.frombuffer(embed_bin_path.read_bytes(),
                                  dtype=np.float16)
    embed_weight = embed_weight.reshape(cfg.vocab_size, cfg.hidden_size)
    body_mlms = [ct.models.MLModel(str(p),
                                    compute_units=ct.ComputeUnit.CPU_AND_NE)
                 for p in body_paths]
    head_mlm = ct.models.MLModel(str(head_path),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pi, prompt in enumerate(PROMPTS):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.input_ids[0].tolist()
        if len(ids) > args.max_seq - args.max_new - 1:
            ids = ids[:args.max_seq - args.max_new - 1]
        print(f"\n=== prompt[{pi}]: {prompt!r} (len={len(ids)}) ===")
        t0 = time.time()
        generated = run_chunked_decode(
            embed_weight, body_mlms, head_mlm, cfg,
            args.max_seq, ids, args.max_new)
        dt = time.time() - t0
        gen_text = tok.decode(generated, skip_special_tokens=False)
        full_text = tok.decode(ids + generated, skip_special_tokens=False)
        print(f"  generated {len(generated)} tokens in {dt:.1f}s "
              f"({len(generated)/dt:.1f} tok/s)")
        print(f"  full: {full_text!r}")
        print(f"  gen:  {gen_text!r}")


if __name__ == "__main__":
    main()
