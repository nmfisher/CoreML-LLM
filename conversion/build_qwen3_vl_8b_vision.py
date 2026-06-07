"""Qwen3-VL 8B vision encoder — thin fork of build_qwen3_vl_2b_vision.py.

Everything in the 2B converter is config-driven (depth, hidden, out_hidden,
patch, deepstack indexes all read from the vision_config), so retargeting is
a single MODEL_ID override. 8B's vision tower is a bigger ViT (hidden 1152,
depth 27, deepstack [8,16,24], merger out = 4096) but the same structure.
Output lands under qwen3_vl_8b_vision/ (bundle dir derives from MODEL_ID).

Usage:
  python build_qwen3_vl_8b_vision.py --out-dir /tmp/qwen3vl8b_stateful --nbits 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_qwen3_vl_2b_vision as V

V.MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


if __name__ == "__main__":
    V.main()
