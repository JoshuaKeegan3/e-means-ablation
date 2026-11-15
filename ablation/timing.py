"""Clean wall-clock comparison: library e-means vs faiss k-means.

Run alone on an idle machine, one job at a time. For each dataset and seed:
  * NumpyEmeans.train (vendored reference, library defaults) at T epochs
  * faiss.Kmeans (same init, no subsampling) at N iterations
Records wall time and held-out MSE to results/timing.jsonl.

  python timing.py [--datasets a,b] [--seeds 3]
"""
import argparse
import json
import os
import time

import numpy as np

from datasets import DATASETS, load
from variants import NumpyEmeans, lloyd_faiss, mse

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "timing.jsonl")
EM_EPOCHS = [1, 2, 3, 5, 10]
KM_ITERS = [1, 2, 3, 5, 10, 20, 30, 50]


def main():
    import faiss

    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--append", action="store_true")
    p.add_argument("--only", default=None, help="run only this algo")
    a = p.parse_args()
    out = open(OUT, "a" if a.append else "w")
    print("faiss %s, %d OpenMP threads" % (faiss.__version__, faiss.omp_get_max_threads()))
    for name in a.datasets.split(","):
        xt, xq, k = load(name)
        n, d = xt.shape
        for seed in range(a.seeds):
            init = xt[np.random.RandomState(seed).permutation(n)[:k]]
            jobs = [("e-means", T) for T in EM_EPOCHS] + [("kmeans", N) for N in KM_ITERS]
            if name == "covtype":
                # revival breaks on this heavy-tailed data; time the
                # library's own switch for turning it off as well
                jobs += [("e-means unused_threshold=0", T) for T in EM_EPOCHS]
            for algo, T in jobs:
                if a.only and algo != a.only:
                    continue
                if algo.startswith("e-means"):
                    em = NumpyEmeans(d, k, nepochs=T, seed=seed,
                                     unused_threshold=0 if "=0" in algo else None)
                    t0 = time.perf_counter()
                    c, _ = em.train(xt, init_centroids=init)
                    t = time.perf_counter() - t0
                else:
                    c, t = lloyd_faiss(xt, k, T, seed, init_centroids=init)
                r = dict(dataset=name, seed=seed, algo=algo, epochs=T, time=t,
                         test_mse=mse(xq, c), n=n, d=d, k=k)
                out.write(json.dumps(r) + "\n")
                out.flush()
                print("%-15s seed=%d %-8s T=%-3d %7.2fs  mse=%.6g"
                      % (name, seed, algo, T, t, r["test_mse"]), flush=True)


if __name__ == "__main__":
    main()
