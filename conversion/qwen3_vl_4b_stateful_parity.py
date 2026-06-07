"""Qwen3-VL 4B stateful Mac parity — thin fork of the 8B stateful parity.

Drives the MLState chunks from `build_qwen3_vl_4b_stateful_chunks.py`.
Retargets the shared 8B harness to the 4B model id (both the builder
module's MODEL_ID, read by load_text_config, and the parity module's
own MODEL_ID copy, used for the tokenizer).

Usage:
  python qwen3_vl_4b_stateful_parity.py \\
      --chunks-dir /tmp/qwen3vl4b_stateful/qwen3_vl_4b_stateful_chunks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_qwen3_vl_8b_stateful_chunks as S
S.MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"      # load_text_config reads this

import qwen3_vl_8b_stateful_parity as P
P.MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"      # tokenizer id in P.main()


if __name__ == "__main__":
    P.main()
