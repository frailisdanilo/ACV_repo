"""
Check a Stage-2 implementation against the authors' kept tokens, on the CPU.

For every captured image and every batch size that was captured, runs the chosen implementation from cluster/pruner_core.py on the authors' exact pivots/similarity/spatial matrices and compares the kept set with the authors' result.
"""
import argparse
import glob
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pruner_core import IMPLEMENTATIONS  


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--impl", default="paper", choices=list(IMPLEMENTATIONS))
    ap.add_argument("--B", default=None, help="batch sizes to test (default: all captured)")
    ap.add_argument("--repeat", type=int, default=3, help="timing repeats per image")
    args = ap.parse_args()

    common = torch.load(os.path.join(args.capture_dir, "common.pt"))
    files = sorted(glob.glob(os.path.join(args.capture_dir, "img_*.pt")))
    if not files:
        raise SystemExit(f"no img_*.pt files in {args.capture_dir}")
    spatial = common["spatial"].numpy().astype(np.float32)
    batches = [int(b) for b in args.B.split(",")] if args.B else common["batches"]
    impl = IMPLEMENTATIONS[args.impl]
    print(f"{len(files)} images | N={common['N']} R={common['R']} tau0={common['threshold']} "
          f"| impl={args.impl} | B={batches}")

    for B in batches:
        exact, sizes, jac, times, gpu_ms, mismatches = 0, [], [], [], [], []
        for f in files:
            d = torch.load(f)
            if B not in d["refs"]:
                continue
            ref = d["refs"][B]["kept"].tolist()
            sim = d["sim"].numpy()                       # fp16, converted inside the implementation
            piv = d["pivots"].tolist()
            ts = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                kept = impl(sim, spatial, piv, common["R"], tau0=common["threshold"], B=B,
                            nf=common["norm_factor"], lam=common["lam"])
                ts.append(1000 * (time.perf_counter() - t))
            times.append(min(ts))
            gpu_ms.append(d["refs"][B]["gpu_ms"])
            sizes.append((len(ref), len(set(kept))))
            j = jaccard(kept, ref)
            jac.append(j)
            if set(kept) == set(ref):
                exact += 1
            else:
                mismatches.append((os.path.basename(f), len(set(kept) - set(ref)), len(set(ref) - set(kept))))
        n = len(jac)
        print(f"\n[B={B}] identical kept set: {exact}/{n} | mean Jaccard {statistics.mean(jac):.4f}"
              f" | kept sizes (authors, impl): {sorted(set(sizes))}")
        print(f"        CPU time (impl) median {statistics.median(times):.2f} ms | "
              f"authors' GPU function (all stages, incl. one-off warm-up) median {statistics.median(gpu_ms):.2f} ms")
        for name, extra, missing in mismatches[:10]:
            print(f"        mismatch {name}: {extra} tokens only in impl, {missing} only in authors'")

    # how different are the authors' own B=1 and B=16 sets?
    if 1 in batches and 16 in batches:
        jj = []
        for f in files:
            d = torch.load(f)
            if 1 in d["refs"] and 16 in d["refs"]:
                jj.append(jaccard(d["refs"][1]["kept"].tolist(), d["refs"][16]["kept"].tolist()))
        print(f"\nauthors' B=1 vs B=16 kept sets: mean Jaccard {statistics.mean(jj):.3f} "
              f"(min {min(jj):.3f}) over {len(jj)} images")


if __name__ == "__main__":
    main()
