"""
Stage 2 token selection of VLMPruner, model independent and on CPU (NumPy)

Inputs:
  sim      [N, N] float  cosine similarity over the top-q channels 
  spatial  [N, N] float  distance between token grid positions 
  pivots   [P]    int    Stage-1 pivots 
  R        int           number of tokens to keep
  tau0, B, nf, lam       initial threshold, token batch size, 1/(grid diagonal), spatial weight (0.5)
Output: list of kept token indices (0..N-1), in the order they were selected.

paper_stage2() is a port of the author's Stage 2, so it should reproduce the exact kept set.
"""

import numpy as np



def _score(sim_rows, dist_rows, nf, lam):
    spatial_reward = dist_rows.min(axis=1, keepdims=True) * np.float32(nf)
    return sim_rows * (np.float32(1.0) + np.float32(lam) * spatial_reward)

def paper_stage2(sim, spatial, pivots, R, tau0=0.8, B=16, nf=1.0/(24*2**0.5), lam=0.5):
    sim = np.asarray(sim, dtype=np.float32)
    spatial = np.asarray(spatial, dtype=np.float32)
    N = sim.shape[0]
    selected = [int(p) for p in pivots]
    threshold = tau0 - 0.1
    pre_num = 0

    while len(selected) < R:
        if pre_num == len(selected):
            break
        pre_num = len(selected)
        threshold = threshold + 0.1
        thr32 = np.float32(threshold)
        mask = np.ones(N, dtype=bool)
        mask[selected] = False
        candidates = np.arange(N)[mask]
        
        if len(candidates) == 0:
            break

        sel = np.array(selected)
        scores = _score(sim[np.ix_(candidates, sel)], spatial[np.ix_(candidates, sel)], nf, lam)
        nondup = np.float32(1.0) - scores.max(axis=1)
        order = np.argsort(-nondup, kind="stable")
        sorted_candidates = candidates[order]

        for i in range(0, len(order), B):
            if i==B and pre_num==len(selected):
                break
            batch = sorted_candidates[i:i+B]
            sel = np.array(selected)
            max_sims = _score(sim[np.ix_(batch, sel)], spatial[np.ix_(batch, sel)], nf, lam).max(axis=1)
            valid = batch[max_sims < thr32]

            if len(selected) + len(valid) > R:
                room = R-len(selected)
                top = np.argsort(-(np.float32(1.0) - max_sims), kind="stable")[:room] #torch.topk
                valid = batch[top]
            if len(valid) > 0:
                selected.extend(int(v) for v in valid)
                if len(selected) >= R:
                    break
    return selected


IMPLEMENTATIONS = {"paper": paper_stage2}
