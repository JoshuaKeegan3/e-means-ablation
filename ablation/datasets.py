"""Benchmark datasets: (train, test, k). All float32, train/test disjoint."""
import numpy as np


def _split(x, n_test, seed=0):
    perm = np.random.RandomState(seed).permutation(len(x))
    x = np.ascontiguousarray(x[perm], dtype="float32")
    return x[n_test:], x[:n_test]


def gmm(n, d, ncomp, seed=0):
    """Gaussian mixture with unequal weights and per-component scales."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(ncomp, d) * 4
    weights = rng.dirichlet(np.ones(ncomp) * 0.5)
    scales = rng.uniform(0.3, 2.0, size=ncomp)
    comp = rng.choice(ncomp, size=n, p=weights)
    return centers[comp] + rng.randn(n, d) * scales[comp, None]


def load(name):
    if name == "gmm-d32":
        return (*_split(gmm(250_000, 32, 2000), 50_000), 1024)
    if name == "gmm-d2":
        return (*_split(gmm(250_000, 2, 500), 50_000), 256)
    if name == "uniform-d64":
        x = np.random.RandomState(0).rand(250_000, 64)
        return (*_split(x, 50_000), 1024)
    if name == "covtype":
        from sklearn.datasets import fetch_covtype

        x = fetch_covtype().data.astype("float64")
        # z-scoring the 44 one-hot columns turns rare categories into
        # coordinates of up to ~380: a heavy-tailed dataset
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        return (*_split(x, 81_012), 1024)
    if name == "covtype-minmax":
        from sklearn.datasets import fetch_covtype

        x = fetch_covtype().data.astype("float64")
        x = (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-12)
        return (*_split(x, 81_012), 1024)
    if name == "mnist":
        from sklearn.datasets import fetch_openml

        x = fetch_openml("mnist_784", version=1, as_frame=False,
                         parser="liac-arff").data / 255.0
        return (*_split(x, 10_000), 256)
    raise KeyError(name)


DATASETS = ["gmm-d32", "uniform-d64", "gmm-d2", "covtype", "covtype-minmax", "mnist"]
