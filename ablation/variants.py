"""Ablatable e-means.

`AblatedEmeans` subclasses the reference `NumpyEmeans` from
faiss/contrib/e_means (vendored, unmodified, in ../vendor/e_means) and lets
each ingredient of e-means be switched off independently. With every switch
on it reproduces `NumpyEmeans.train` exactly (see check_equivalence.py).

Ingredients (Algorithm 1 of the e-means paper, arXiv:2611.09501):

  revival        every 5 updates, split heavy clusters into ones whose EMA mass
                 fell below 0.1 x the mean mass (line 9 / Appendix J)
  closing        return the unweighted mean of the points assigned during the
                 last epoch, instead of the raw EMA positions (line 12)
  cosine         cosine-anneal the EMA step alpha from alpha0 to 0.1*alpha0
                 (Eq. 5). Off = constant alpha0
  ema            exponential moving average of per-cluster sums/counts with
                 decay gamma = 1 - alpha (Eq. 2). Off = Sculley's mini-batch
                 k-means: counts accumulate forever, per-cluster step n_j/N_j
                 (no forgetting; alpha/schedule/step law then have no effect)
  batch_law      batch size B from the fitted scaling law (Eq. 7). Off = a
                 fixed B=1024 (scikit-learn's MiniBatchKMeans default); the
                 step law still sets alpha0 for that B
  step_law       alpha0 from the fitted step law (Eq. 6). Off = alpha0 = B/n,
                 i.e. an EMA whose memory is about one epoch
  shuffle        a fresh random permutation each epoch. Off = one permutation
                 drawn once and reused every epoch
  minibatch      off = one full-batch update per epoch with alpha=1 and the
                 old mass discarded: Lloyd's k-means. Implies revival,
                 closing, cosine and EMA have no effect.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))

from e_means import NumpyEmeans  # noqa: E402
from e_means.base import _epoch_seed, _scheduled_alpha  # noqa: E402

COMPONENTS = (
    "revival",
    "closing",
    "cosine",
    "ema",
    "batch_law",
    "step_law",
    "shuffle",
    "minibatch",
)

FIXED_BATCH = 1024


class AblatedEmeans(NumpyEmeans):
    def __init__(self, d, k, nepochs=5, seed=1234, n_train=None, **switches):
        unknown = set(switches) - set(COMPONENTS)
        if unknown:
            raise ValueError("unknown switches %s" % sorted(unknown))
        self.on = {c: bool(switches.get(c, True)) for c in COMPONENTS}
        on = self.on
        batch_size = alpha0 = None
        if not on["minibatch"]:
            batch_size, alpha0 = n_train, 1.0
        else:
            if not on["batch_law"]:
                batch_size = FIXED_BATCH
            if not on["step_law"]:
                if n_train is None:
                    raise ValueError("step_law=False needs n_train")
                if batch_size is None:
                    # resolve the law's batch without the step law
                    from e_means.base import _resolve_batch_law, _resolve_recipe

                    law, _ = _resolve_batch_law("auto", d)
                    batch_size, _ = _resolve_recipe(
                        n_train, k, float(nepochs), law, None, None
                    )
                alpha0 = min(1.0, batch_size / n_train)
        super().__init__(
            d,
            k,
            nepochs=nepochs,
            batch_size=batch_size,
            alpha0=alpha0,
            seed=seed,
            capture_assignments=False,
            # threshold None -> library default (0.1, or 0 when d <= 2)
            unused_threshold=(
                None if (on["revival"] and on["minibatch"]) else 0.0
            ),
        )

    # -- Eq. 5 cosine schedule ------------------------------------------------
    def _alpha(self):
        if self.on["cosine"] and self.on["minibatch"]:
            return _scheduled_alpha(
                self.recipe["alpha0"], self._step, self.recipe["total_steps"]
            )
        return self.recipe["alpha0"]

    def update(self, stats):
        # Same as _EmeansBase.update, with the schedule and the EMA rule
        # pluggable.
        self._require_running()
        expected_n = min(
            self.recipe["batch_size"],
            self.recipe["epoch_rows"][self._epoch] - self._epoch_rows,
        )
        ids, sums, counts, objective = self._prepare_statistics(
            stats, expected_n
        )
        if not self.on["minibatch"]:
            # Lloyd: centroid = mean of its points, previous mass discarded
            alpha, decay, replace = 1.0, 0.0, True
        elif not self.on["ema"]:
            # Sculley mini-batch k-means: N_j += n_j, step n_j / N_j
            alpha, decay, replace = 1.0, 1.0, False
        else:
            alpha = self._alpha()
            alpha, decay, replace = alpha, 1.0 - alpha, alpha == 1
        self._apply_statistics(
            ids, sums, counts, objective, alpha, decay, replace
        )
        self._finish_step(expected_n)

    # -- line 12 closing means ------------------------------------------------
    def finalize(self):
        if self.on["closing"] and self.on["minibatch"]:
            return super().finalize()
        raw = self._centroids.copy()
        super().finalize()
        self.centroids = raw
        return self.centroids

    # -- training loop (NumpyEmeans._train_rows without label capture) ---------
    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        n = len(x)
        if init_centroids is None:
            init_centroids = x[
                np.random.RandomState(self.seed).permutation(n)[: self.k]
            ]
        t0 = time.perf_counter()
        self.begin(n, init_centroids)
        B = self.recipe["batch_size"]
        for epoch, rows in enumerate(self.recipe["epoch_rows"]):
            seed_epoch = epoch if self.on["shuffle"] else 0
            order = np.random.RandomState(
                _epoch_seed(self.seed, seed_epoch)
            ).permutation(n)
            for begin in range(0, rows, B):
                batch = order[begin : min(begin + B, rows)]
                self.update(self._accumulate(x, batch))
            self.end_epoch()
        self.finalize()
        self.train_time = time.perf_counter() - t0
        return self.centroids


def lloyd_faiss(x, k, niter, seed, init_centroids=None):
    """faiss.Kmeans (Lloyd) with subsampling off, for the k-means baseline."""
    import faiss

    km = faiss.Kmeans(
        x.shape[1],
        k,
        niter=niter,
        seed=seed,
        max_points_per_centroid=10**9,
        min_points_per_centroid=1,
        verbose=False,
    )
    t0 = time.perf_counter()
    km.train(np.ascontiguousarray(x, dtype="float32"), init_centroids=init_centroids)
    return km.centroids, time.perf_counter() - t0


def mse(x, centroids):
    import faiss

    D, _ = faiss.knn(np.ascontiguousarray(x, dtype="float32"), centroids, 1)
    return float(D.astype("float64").mean())
