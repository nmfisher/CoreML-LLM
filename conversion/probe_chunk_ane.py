#!/usr/bin/env python3
"""Isolated ANE-acceptance probe for a single compiled-at-build .mlpackage.

Loads the package with a chosen compute unit (CPU_ONLY vs CPU_AND_NE), feeds
zero-valued inputs shaped exactly to the model spec, warms up, times predicts.

Interpretation (calibrated by CONTROLS run through the same harness):
  - chunk1_3way (positive, known ANE-engaged in Swift at 14ms vs 189ms CPU):
      CPU_AND_NE predict must be MUCH faster than CPU_ONLY. If it isn't, this
      harness is silently falling back and the timing proxy is void.
  - chunk2_3way (negative, known error -1 in Swift): see whether coremltools
      raises (faithful -1) or silently runs at CPU speed (fallback).
  - partials: CPU_AND_NE faster-than-CPU  => ANE ACCEPTS (size ok).
              CPU_AND_NE ~ CPU speed       => ANE REJECTS (fell back).
              CPU_AND_NE raises            => ANE REJECTS (hard -1).
"""
import argparse, os, time
import numpy as np
import coremltools as ct

# All chunk inputs are built dtype=fp16 (see build_gemma4_3way.build_chunk*),
# so the spec is fp16 throughout; we feed float16 zeros sized to the spec.
def _spec_inputs(mlmodel):
    """Return {name: shape} from the MLProgram spec (symbolic dims -> 1)."""
    spec = mlmodel.get_spec()
    out = {}
    for inp in spec.description.input:
        ma = inp.type.multiArrayType
        shape = tuple(int(d) if isinstance(d, int) else 1 for d in ma.shape) or (1,)
        out[inp.name] = shape
    return out

def _load(path, compute_units):
    return ct.models.MLModel(path, compute_units=compute_units)

def _bench(path, compute_units, warm=2, iters=5):
    try:
        m = _load(path, compute_units)
    except Exception as e:
        return ("LOAD_ERR", str(e)[:160])
    ins = _spec_inputs(m)
    feed = {n: np.zeros(shape, dtype=np.float16) for n, shape in ins.items()}
    # warmup (first call compiles for the chosen unit)
    try:
        for _ in range(warm):
            m.predict(feed)
    except Exception as e:
        return ("PREDICT_ERR", str(e)[:200])
    t0 = time.time()
    for _ in range(iters):
        m.predict(feed)
    ms = (time.time() - t0) / iters * 1000
    return ("OK", ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+", help=".mlpackage paths")
    ap.add_argument("--no-ne", action="store_true", help="skip CPU_AND_NE")
    args = ap.parse_args()

    print(f"{'package':<46} {'CPU_ONLY':>14} {'CPU_AND_NE':>20} {'NE/CPU':>8}")
    print("-" * 92)
    for p in args.packages:
        label = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        label = label[:44]
        cpu = _bench(p, ct.ComputeUnit.CPU_ONLY)
        ne = ("skip", "-") if args.no_ne else _bench(p, ct.ComputeUnit.CPU_AND_NE)
        cpu_s = f"{cpu[1]:.1f}ms" if cpu[0] == "OK" else f"{cpu[0]}"
        if ne[0] == "OK":
            ne_s = f"{ne[1]:.1f}ms"
            ratio = f"{cpu[1]/ne[1]:.2f}" if cpu[0]=="OK" else "-"
        elif ne[0] in ("LOAD_ERR", "PREDICT_ERR"):
            ne_s = f"{ne[0]}"
            ratio = "REJECT?"
        else:
            ne_s = str(ne[1]); ratio = "-"
        print(f"{label:<46} {cpu_s:>14} {ne_s:>20} {ratio:>8}")

if __name__ == "__main__":
    main()
