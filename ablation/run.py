"""Run the ablation grid. Appends one JSON line per run to results/runs-<dataset>.jsonl
(already-finished runs are skipped, so it can be resumed).

  python run.py [--datasets a,b] [--seeds 3] [--epochs 1,2,3,5,10]
"""
import argparse
import json
import os
import time

import numpy as np

from datasets import DATASETS, load
from variants import AblatedEmeans, lloyd_faiss, mse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Cumulative strip-down from e-means to Lloyd's k-means, one ingredient per rung.
LADDER = [
    ("L0 e-means", {}),
    ("L1 - revival", dict(revival=False)),
    ("L2 - closing means", dict(revival=False, closing=False)),
    ("L3 - cosine schedule", dict(revival=False, closing=False, cosine=False)),
    ("L4 - EMA (= Sculley mini-batch)",
     dict(revival=False, closing=False, cosine=False, ema=False)),
    ("L5 - mini-batch (= Lloyd)",
     dict(revival=False, closing=False, cosine=False, ema=False, minibatch=False)),
]

# Remove exactly one ingredient from full e-means.
LOO = [
    ("-revival", dict(revival=False)),
    ("-closing", dict(closing=False)),
    ("-cosine", dict(cosine=False)),
    ("-ema (Sculley)", dict(ema=False)),
    ("-batch_law (B=1024)", dict(batch_law=False)),
    ("-step_law (alpha0=B/n)", dict(step_law=False)),
    ("-shuffle", dict(shuffle=False)),
]

LLOYD_CONVERGED = 50


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            keys.add((r["dataset"], r["seed"], r["epochs"], r["variant"]))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", default="1,2,3,5,10")
    a = p.parse_args()
    epochs = [float(e) for e in a.epochs.split(",")]
    os.makedirs(RESULTS, exist_ok=True)
    out = None

    def record(**r):
        out.write(json.dumps(r) + "\n")
        out.flush()
        print("  %-12s seed=%d T=%-4g %-34s test=%.6g  %.1fs"
              % (r["dataset"], r["seed"], r["epochs"], r["variant"],
                 r["test_mse"], r["time"]), flush=True)

    for name in a.datasets.split(","):
        path = os.path.join(RESULTS, "runs-%s.jsonl" % name)
        done = done_keys(path)
        if out:
            out.close()
        out = open(path, "a")
        xt, xq, k = load(name)
        n, d = xt.shape
        print("%s: n=%d d=%d k=%d test=%d" % (name, n, d, k, len(xq)), flush=True)
        for seed in range(a.seeds):
            init = xt[np.random.RandomState(seed).permutation(n)[:k]]
            base = dict(dataset=name, seed=seed, n=n, d=d, k=k)
            jobs = [(T, "kmeans (faiss)", None) for T in epochs + [LLOYD_CONVERGED]]
            for T in epochs:
                seen = set()
                for group, specs in (("ladder", LADDER), ("loo", LOO)):
                    for label, sw in specs:
                        key = tuple(sorted(sw.items()))
                        if key not in seen:  # L1 == -revival
                            seen.add(key)
                            jobs.append((T, label, sw))
            for T, label, sw in jobs:
                if (name, seed, T, label) in done:
                    continue
                if sw is None:
                    c, t = lloyd_faiss(xt, k, int(T), seed, init_centroids=init)
                    recipe = {}
                else:
                    m = AblatedEmeans(d, k, nepochs=T, seed=seed, n_train=n, **sw)
                    c = m.fit(xt, init)
                    t = m.train_time
                    recipe = dict(batch_size=m.recipe["batch_size"],
                                  alpha0=m.recipe["alpha0"],
                                  unused_threshold=m.recipe["unused_threshold"],
                                  n_splits=int(m.n_splits))
                record(**base, epochs=T, variant=label, switches=sw or {},
                       train_mse=mse(xt, c), test_mse=mse(xq, c), time=t,
                       **recipe)
    out.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("total %.0fs" % (time.time() - t0))
