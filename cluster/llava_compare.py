"""Compare finished LLaVA eval runs with each other and with the paper.

Prints
  1. scores vs the paper (POPE F1 = mean over the 3 categories; OCRBench = correct out of 1000)
  2. paired differences between methods with 95% bootstrap confidence intervals:
     the same questions are resampled for both methods, so question difficulty cancels out
  3. how many questions each method got right where the other got it wrong
  4. median time per question and selection time 
"""
import argparse
import glob
import json
import os

import numpy as np

PAPER = {  # LLaVA-1.5-7B, 64 tokens (Tables 6 and 11); unpruned = "Vanilla"
    "full": {"pope": 85.9, "ocrbench": 297},
    "B1": {"pope": 83.6, "ocrbench": 275},
    "B16": {"pope": 82.2, "ocrbench": 279},
}
ORDER = ["full", "B1", "B16", "ours"]


def latest_runs():
    runs = {}
    for d in glob.glob("runs/llava_eval/*"):
        s = os.path.join(d, "summary.json")
        if os.path.exists(s):
            m = json.load(open(s))["method"]
            if m not in runs or os.path.getmtime(s) > os.path.getmtime(os.path.join(runs[m], "summary.json")):
                runs[m] = d
    return runs


def load(run, bench):
    path = os.path.join(run, f"{bench}_answers.jsonl")
    if not os.path.exists(path):
        return None
    return {r["id"]: r for r in map(json.loads, open(path))}


def pope_f1_boot(recs, ids_by_cat, idx_by_cat):
    """mean over categories of F1, for each bootstrap resample. Returns array [B]."""
    f1s = []
    for cat, ids in ids_by_cat.items():
        p = np.array([recs[i]["pred"] == "yes" for i in ids])
        g = np.array([recs[i]["gt"] == "yes" for i in ids])
        P, G = p[idx_by_cat[cat]], g[idx_by_cat[cat]]
        tp = (P & G).sum(1); fp = (P & ~G).sum(1); fn = (~P & G).sum(1)
        f1s.append(100 * 2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return np.mean(f1s, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run folders (default: latest finished run per method)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    runs = {json.load(open(os.path.join(r, "summary.json")))["method"]: r for r in args.runs} if args.runs else latest_runs()
    methods = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    if not methods:
        raise SystemExit("no finished runs (runs/llava_eval/*/summary.json) found")
    summ = {m: json.load(open(os.path.join(runs[m], "summary.json"))) for m in methods}
    rng = np.random.default_rng(0)

    print("runs:")
    for m in methods:
        print(f"  {m:5s} {runs[m]}  (commit {summ[m]['env'].get('commit')}, {summ[m]['env'].get('gpu')})")

    # 1. scores vs paper
    print("\n1) scores vs paper")
    print(f"  {'method':6s} {'POPE F1':>8s} {'paper':>6s} {'OCRBench':>9s} {'paper':>6s}")
    for m in methods:
        p = summ[m].get("pope", {}).get("f1_mean")
        o = summ[m].get("ocrbench", {}).get("score")
        pp, po = PAPER.get(m, {}).get("pope"), PAPER.get(m, {}).get("ocrbench")
        fmt = lambda v, f: (f % v) if v is not None else "-"
        print(f"  {m:6s} {fmt(p, '%8.2f')} {fmt(pp, '%6.1f')} {fmt(o, '%9d')} {fmt(po, '%6d')}")

    # 2./3. paired comparisons
    print(f"\n2) paired differences (row minus column), 95% bootstrap CI over {args.boot} resamples")
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    for bench in ("pope", "ocrbench"):
        data = {m: load(runs[m], bench) for m in methods}
        data = {m: d for m, d in data.items() if d}
        if len(data) < 2:
            continue
        print(f"  [{bench}]")
        for a, b in pairs:
            if a not in data or b not in data:
                continue
            ids = sorted(set(data[a]) & set(data[b]))
            if bench == "pope":
                cats = sorted({data[a][i]["group"] for i in ids})
                ids_by_cat = {c: [i for i in ids if data[a][i]["group"] == c] for c in cats}
                idx = {c: rng.integers(0, len(v), size=(args.boot, len(v))) for c, v in ids_by_cat.items()}
                full_idx = {c: np.arange(len(v))[None, :] for c, v in ids_by_cat.items()}
                point = (pope_f1_boot(data[a], ids_by_cat, full_idx) - pope_f1_boot(data[b], ids_by_cat, full_idx))[0]
                diff = pope_f1_boot(data[a], ids_by_cat, idx) - pope_f1_boot(data[b], ids_by_cat, idx)
                unit = "F1 points"
            else:
                ca = np.array([data[a][i]["correct"] for i in ids], dtype=float)
                cb = np.array([data[b][i]["correct"] for i in ids], dtype=float)
                point = ca.sum() - cb.sum()
                idx = rng.integers(0, len(ids), size=(args.boot, len(ids)))
                diff = ca[idx].sum(1) - cb[idx].sum(1)
                unit = "points"
            lo, hi = np.percentile(diff, [2.5, 97.5])
            star = " *" if lo > 0 or hi < 0 else ""
            only_a = sum(data[a][i]["correct"] and not data[b][i]["correct"] for i in ids)
            only_b = sum(data[b][i]["correct"] and not data[a][i]["correct"] for i in ids)
            print(f"    {a:5s} - {b:5s}: {point:+7.2f} {unit}  [{lo:+.2f}, {hi:+.2f}]{star}"
                  f"   (n={len(ids)}; right only in {a}: {only_a}, only in {b}: {only_b})")
    print("  * = interval excludes 0")

    # 4. timing
    print("\n4) median time per question (indicative; quality runs may share the node)")
    for bench in ("pope", "ocrbench"):
        print(f"  [{bench}]")
        base = summ.get("full", {}).get(bench, {}).get("time_ms_median")
        for m in methods:
            r = summ[m].get(bench)
            if not r:
                continue
            rel = f"  x{base / r['time_ms_median']:.2f} vs full" if base else ""
            print(f"    {m:5s} total {r['time_ms_median']:7.1f} ms | selection {r['select_ms_median']:6.1f} ms"
                  f" | seq {r['seq_in_median']:.0f} -> {r['seq_after_median']:.0f}{rel}")


if __name__ == "__main__":
    main()
