"""Qwen3-VL 4B chunk_0_vision — thin fork of the 8B chunk_0_vision builder.

Sets MODEL_ID to the 4B; layers_per_chunk (6), hidden (2560), tied head and
the output subdir (qwen3_vl_4b_stateful_chunks) all derive from config.

Usage:
  python build_qwen3_vl_4b_stateful_chunk0_vision.py \\
      --out-dir /tmp/qwen3vl4b_stateful --num-chunks 6 --nbits 4 --group-size 64
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_qwen3_vl_8b_stateful_chunks as S
S.MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

import build_qwen3_vl_8b_stateful_chunk0_vision as V0


if __name__ == "__main__":
    V0.main()
