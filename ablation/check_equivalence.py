"""Sanity checks for the ablation harness.

1. All switches on == reference NumpyEmeans.train, bit for bit.
2. minibatch=False == Lloyd's k-means: matches faiss.Kmeans from the same
   initial centroids (up to float summation order) whenever no cluster
   goes empty (faiss re-seeds empty clusters, plain Lloyd does not).
"""
import numpy as np

from variants import AblatedEmeans, NumpyEmeans, lloyd_faiss, mse


def main():
    rng = np.random.RandomState(0)
    for d, k, n in [(32, 100, 20000), (2, 50, 10000), (64, 256, 30000), (16, 64, 50000)]:
        centers = rng.randn(k, d) * 3
        x = (centers[rng.randint(k, size=n)] + rng.randn(n, d)).astype("float32")
        ref = NumpyEmeans(d, k, nepochs=3, seed=7)
        c_ref, _ = ref.train(x)
        ab = AblatedEmeans(d, k, nepochs=3, seed=7, n_train=n)
        c_ab = ab.fit(x)
        assert ref.recipe == ab.recipe, "recipe mismatch"
        assert np.array_equal(c_ref, c_ab), "centroids differ"
        print("d=%d k=%d: AblatedEmeans(all on) == NumpyEmeans  OK  (B=%d alpha0=%.3g)"
              % (d, k, ab.recipe["batch_size"], ab.recipe["alpha0"]))

        init = x[np.random.RandomState(7).permutation(n)[:k]]
        ll = AblatedEmeans(d, k, nepochs=10, seed=7, n_train=n, minibatch=False)
        c_ll = ll.fit(x, init)
        c_fk, _ = lloyd_faiss(x, k, 10, 7, init_centroids=init)
        a, b = mse(x, c_ll), mse(x, c_fk)
        import faiss
        _, lab = faiss.knn(x, c_ll, 1)
        empty = k - len(np.unique(lab))
        print("        Lloyd rung MSE %.6f vs faiss.Kmeans %.6f (rel diff %.2e, "
              "%d empty clusters)" % (a, b, abs(a - b) / b, empty))
        # faiss re-seeds empty clusters by splitting; the pure Lloyd rung
        # does not, so they only agree when no cluster goes empty.
        if empty == 0:
            assert abs(a - b) / b < 1e-3


if __name__ == "__main__":
    main()
