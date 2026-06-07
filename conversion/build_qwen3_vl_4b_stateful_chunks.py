"""Qwen3-VL 4B stateful decode converter — thin fork of the 8B builder.

Same MLState + slice_update recipe as `build_qwen3_vl_8b_stateful_chunks.py`.
4B differs only in MODEL_ID, which flows through to:
  * 36 layers / 6 chunks (same split as 8B), hidden 2560, intermediate 9728
    — all config-derived.
  * **TIED** lm_head: 4B's top-level config has tie_word_embeddings=True,
    so `load_tie_word_embeddings()` returns True and the head re-uses
    embed_tokens.weight (no separate lm_head). The 8B builder's ANEHeadChunk
    already handles both cases.
  * Output subdir derives from MODEL_ID → qwen3_vl_4b_stateful_chunks/.

INT4 grouped (gs=64) matches the VLMKit MLX `Qwen3-VL-4B-Instruct-MLX-4bit`.

Usage:
  python build_qwen3_vl_4b_stateful_chunks.py \\
      --out-dir /tmp/qwen3vl4b_stateful --num-chunks 6 --nbits 4 --group-size 64
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_qwen3_vl_8b_stateful_chunks as S

# Retarget the shared builder. load_text_config / load_tie_word_embeddings /
# load_text_backbone all read S.MODEL_ID at call time, and main() derives
# the output subdir from it, so this single override is sufficient.
S.MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


if __name__ == "__main__":
    S.main()
