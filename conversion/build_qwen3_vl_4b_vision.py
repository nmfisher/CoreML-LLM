"""Qwen3-VL 4B vision encoder — thin fork of build_qwen3_vl_2b_vision.py.

4B's vision tower is identical to 2B (hidden 1024, depth 24, deepstack
[5,11,17]) except the merger output = text hidden 2560. Config-driven, so
a single MODEL_ID override suffices. Output lands under qwen3_vl_4b_vision/.

Usage:
  python build_qwen3_vl_4b_vision.py --out-dir /tmp/qwen3vl4b_stateful --nbits 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_qwen3_vl_2b_vision as V

V.MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


if __name__ == "__main__":
    V.main()
