# Qwen3-VL 4B & 8B — Core ML stateful (text + vision)

Both sizes ship the full text **and** image-chat path on the ANE, built on
the 2B stateful pipeline. Confirmed working on iPhone 17 Pro (text + image,
both sizes) 2026-06-07.

## Architecture (vs 2B)

| | 2B | 4B | 8B |
|---|---|---|---|
| layers / chunks | 28 / 4 | 36 / 6 | 36 / 6 |
| hidden | 2048 | 2560 | 4096 |
| lm_head | tied | **tied** | **untied** |
| text bundle | 2.3 GB | 2.6 GB | 4.7 GB |
| vision tower | hidden 1024, depth 24, deepstack [5,11,17] | **same as 2B** | hidden 1152, depth 27, deepstack [8,16,24] |
| total (text+vision) | — | ~3.3 GB | ~5.9 GB |

Everything else is config-derived. Quantization: decode chunks INT4
per-grouped-channel (group_size 64, matching the MLX `*-4bit` builds);
vision encoder INT8 (99.8% ANE). ctx 2048, 196 image tokens (448×448).

## Conversion (Mac, `conversion/.venv`)

```bash
P=/path/to/out   # e.g. ~/Downloads/qwen3_vl_8b_coreml
# 1. text decode body + head + embed sidecar (stateful, MLState slice_update)
python build_qwen3_vl_8b_stateful_chunks.py        --out-dir $P --num-chunks 6 --nbits 4 --group-size 64
# 2. DeepStack-aware chunk_0_vision (same KV state as chunk_0)
python build_qwen3_vl_8b_stateful_chunk0_vision.py --out-dir $P --num-chunks 6 --nbits 4 --group-size 64
# 3. vision encoder (ViT + merger + 3 DeepStack taps)
python build_qwen3_vl_8b_vision.py                 --out-dir $P --nbits 8
# parity (Mac CPU+ANE greedy, EN/JP/reasoning):
python qwen3_vl_8b_stateful_parity.py --chunks-dir $P/qwen3_vl_8b_stateful_chunks
```

4B is identical with `4b`/`4B` names. The 4B/8B builders are thin forks of
`build_qwen3_vl_8b_stateful_chunks.py` (just set `MODEL_ID`); the vision
forks set `MODEL_ID` on `build_qwen3_vl_2b_vision.py`. There is also an
I/O-KV path (`build_qwen3_vl_8b_text_decode_chunks_ane.py` + parity) — the
literal "same as the 4B decode" path, used as the correctness gate.

## Sideload + run

```bash
scripts/qwen3vl8b_stateful_push.sh $P/qwen3_vl_8b_stateful_chunks   # chunks + chunk_0_vision + vision encoder
```
Lands in `Documents/Models/qwen3-vl-8b-stateful/{qwen3_vl_8b_stateful_chunks, qwen3_vl_8b_vision}`.
In-app: pick **Qwen3-VL 8B (stateful) + vision** → image button → ask.

## Swift

`Qwen3VL2BStatefulGenerator` is size-agnostic — `Config.default4B` /
`.default8B` carry `chunkSubdir` + `modelDirName`. `LLMRunner` detects /
loads / generates 2B/4B/8B (text + image), reusing the size-agnostic
`Qwen3VL2BVisionEncoder` (`resolveModel(folder:subdir:)`). `ModelDownloader`
has entries + local detection. HF: `mlboydaisuke/qwen3-vl-{4b,8b}-stateful-coreml`.

## Gotchas that cost real time

- **8B untied lm_head**: `tie_word_embeddings` is on the *top-level*
  `Qwen3VLConfig` only (the text sub-config omits it). Read it via
  `AutoConfig`; 8B=False (separate `lm_head.weight`), 4B/2B=True (tied).
- **8B `mrope_interleaved=True`** is a no-op for text (T=H=W=pos ⇒ the
  interleave copies identical values) — standard 1D RoPE is exact.
- **Vision conversion broke on transformers 5.5.0**: the HF vision
  attention's non-flash path does `torch.split(q/k/v, lengths.tolist())`
  (unconvertible int op) and `rotate_half` uses `x[..., :x.shape[-1]//2]`
  (aten::Int). Fixed in `build_qwen3_vl_2b_vision.py` with a single-image
  full-SDPA attention replacement + a `torch.chunk` RoPE. Inherited by the
  4B/8B forks.
- **Disk**: the stateful builder deletes each chunk's fp16 intermediate
  immediately (keeping all 6 peaks at ~14 GB and crashed with
  `MIL FileWriter` on a near-full disk).
- **Never run two conversions writing the same out-dir concurrently** —
  serialize (chain on `pgrep`); a stray `rm -rf` mid-run corrupts both.
- **Xcode project is an explicit file list** (not a synchronized group) —
  new `.swift` files must be registered in `project.pbxproj` (use the
  `xcodeproj` gem) or you get "Cannot find … in scope".
