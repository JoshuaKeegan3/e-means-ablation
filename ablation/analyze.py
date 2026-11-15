"""Turn results/runs.jsonl into results/*.md tables and figures/*.png."""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import LADDER, LLOYD_CONVERGED, LOO  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "results", "runs-*.jsonl")
FIG = os.path.join(ROOT, "figures")
KM = "kmeans (faiss)"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BETTER, WORSE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def load():
    """mean test MSE and its per-seed values, keyed (dataset, T, variant)."""
    vals = defaultdict(dict)
    meta = {}
    import glob

    lines = [l for p in sorted(glob.glob(RUNS)) for l in open(p)]
    for line in lines:
        r = json.loads(line)
        vals[(r["dataset"], r["epochs"], r["variant"])][r["seed"]] = r["test_mse"]
        meta[r["dataset"]] = (r["n"], r["d"], r["k"])
    return vals, meta


def rel(vals, ds, T, variant, ref_T, ref_variant):
    """Mean over seeds of paired relative difference in %, and its std."""
    a, b = vals.get((ds, T, variant)), vals.get((ds, ref_T, ref_variant))
    if not a or not b:
        return None
    seeds = sorted(set(a) & set(b))
    d = np.array([100 * (a[s] / b[s] - 1) for s in seeds])
    return d.mean(), d.std(), len(seeds)


def fmt(x):
    return "—" if x is None else "%+.1f%%" % x[0]


def main():
    vals, meta = load()
    datasets = [d for d in ["gmm-d32", "uniform-d64", "gmm-d2", "covtype", "covtype-minmax", "mnist"] if d in meta]
    epochs = sorted({T for (_, T, v) in vals if v != KM or T != LLOYD_CONVERGED})
    epochs = [T for T in epochs if T != LLOYD_CONVERGED]
    ladder = [l for l, _ in LADDER]
    loo = [l for l, _ in LOO]
    md = []

    md.append("## Datasets\n\n| dataset | n train | d | k |\n|---|---|---|---|")
    for ds in datasets:
        n, d, k = meta[ds]
        md.append("| %s | %d | %d | %d |" % (ds, n, d, k))

    md.append("\n## Ladder: strip e-means down to Lloyd one ingredient at a time\n")
    md.append("Held-out MSE, mean paired %% difference over seeds. "
              "`vs km@T`: against faiss k-means run for the same number of "
              "epochs (negative = e-means variant better). `vs km@%d`: against "
              "k-means at %d iterations (≈ converged)." % (LLOYD_CONVERGED, LLOYD_CONVERGED))
    for T in epochs:
        md.append("\n### T = %g epochs\n" % T)
        md.append("| rung | " + " | ".join("%s vs km@T | vs km@%d" % (ds, LLOYD_CONVERGED) for ds in datasets) + " |")
        md.append("|---|" + "---|---|" * len(datasets))
        for v in ladder:
            cells = []
            for ds in datasets:
                cells += [fmt(rel(vals, ds, T, v, T, KM)),
                          fmt(rel(vals, ds, T, v, LLOYD_CONVERGED, KM))]
            md.append("| %s | %s |" % (v, " | ".join(cells)))
        md.append("\n Step impact (Δ vs previous rung, % of held-out MSE; positive = removing it hurt):\n")
        md.append("| removed | " + " | ".join(datasets) + " |")
        md.append("|---|" + "---|" * len(datasets))
        for prev, v in zip(ladder, ladder[1:]):
            md.append("| %s | %s |" % (v.split(" ", 1)[1].lstrip("- "),
                      " | ".join(fmt(rel(vals, ds, T, v, T, prev)) for ds in datasets)))

    md.append("\n## Leave one out: remove a single ingredient from full e-means\n")
    md.append("Mean paired % change in held-out MSE vs full e-means at the same "
              "T (positive = the ingredient helps). Last row: full e-means vs "
              "faiss k-means at the same T.\n")
    for T in epochs:
        md.append("\n### T = %g epochs\n" % T)
        md.append("| removed | " + " | ".join(datasets) + " |")
        md.append("|---|" + "---|" * len(datasets))
        for v in loo:
            label = v if v != "-revival" else "-revival"
            variant = v if v != "-revival" else "L1 - revival"
            md.append("| %s | %s |" % (label, " | ".join(
                fmt(rel(vals, ds, T, variant, T, "L0 e-means")) for ds in datasets)))
        md.append("| *e-means vs k-means@T* | %s |" % " | ".join(
            fmt(rel(vals, ds, T, "L0 e-means", T, KM)) for ds in datasets))

    open(os.path.join(ROOT, "results", "tables.md"), "w").write("\n".join(md) + "\n")
    os.makedirs(FIG, exist_ok=True)
    plot_ladder(vals, datasets, epochs, ladder)
    plot_loo(vals, datasets, loo, [T for T in (1.0, 5.0, 10.0) if T in epochs])
    plot_curves(vals, datasets, epochs, ladder)


def short(v):
    return {"L0 e-means": "e-means", "L1 - revival": "− revival",
            "L2 - closing means": "− closing", "L3 - cosine schedule": "− cosine",
            "L4 - EMA (= Sculley mini-batch)": "− EMA\n(Sculley)",
            "L5 - mini-batch (= Lloyd)": "− mini-batch\n(Lloyd)"}.get(v, v)


def grid(n):
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


PCT_TICKS = [-50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 200, 400]


def pct_axis(ax, which="y"):
    """symlog % axis with plain, signed tick labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    getattr(ax, "set_%sscale" % which)("symlog", linthresh=10, linscale=1.5)
    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if which == "y" else ax.get_xlim())
    ticks = PCT_TICKS if hi - lo > 30 else sorted(PCT_TICKS + [-2, -1, 1, 2])
    axis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: "%+g%%" % v if v else "0"))


def plot_ladder(vals, datasets, epochs, ladder):
    ramp = ["#b7d3f2", "#82b2e8", "#4f8fdc", "#2a6fbf", "#1d4f8c", "#12325a"]
    fig, axes = grid(len(datasets))
    x = np.arange(len(ladder))
    for ax, ds in zip(axes, datasets):
        ax.axhline(0, color=INK2, lw=1, ls="--")
        for i, T in enumerate(epochs):
            ys = [rel(vals, ds, T, v, LLOYD_CONVERGED, KM) for v in ladder]
            ys = [np.nan if y is None else y[0] for y in ys]
            ax.plot(x, ys, "-o", color=ramp[i % len(ramp)], lw=2, ms=4,
                    label="T=%g" % T)
        ax.set_xticks(x, [short(v) for v in ladder], fontsize=7)
        ax.set_title(ds)
        ax.set_ylabel("held-out MSE vs k-means@%d" % LLOYD_CONVERGED)
        pct_axis(ax)
    axes[0].legend(frameon=False, title="epochs", fontsize=8)
    fig.suptitle("Removing e-means ingredients one at a time, cumulatively "
                 "(rightmost rung = Lloyd's k-means)", fontsize=11)
    fig.text(0.5, 0.005, "Dashed line = Lloyd's k-means after %d iterations; "
             "below it = better than converged k-means. Mean of 3 seeds."
             % LLOYD_CONVERGED, ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(os.path.join(FIG, "ladder.png"), dpi=150)
    plt.close(fig)


def plot_loo(vals, datasets, loo, Ts):
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

    cmap = LinearSegmentedColormap.from_list(
        "div", [BETTER, "#dcdbd6", WORSE])
    norm = SymLogNorm(linthresh=2, vmin=-60, vmax=60)
    rows = [("removed: " + v.lstrip("-"), "L1 - revival" if v == "-revival" else v, "L0 e-means")
            for v in loo]
    rows.append(("e-means vs k-means (same T)", "L0 e-means", KM))
    fig, axes = plt.subplots(1, len(Ts), figsize=(4.6 * len(Ts) + 1.2, 4.2),
                             squeeze=False)
    for ax, T in zip(axes[0], Ts):
        M = np.full((len(rows), len(datasets)), np.nan)
        for i, (_, v, ref) in enumerate(rows):
            for j, ds in enumerate(datasets):
                r = rel(vals, ds, T, v, T, ref)
                if r:
                    M[i, j] = r[0]
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, "%+.1f" % M[i, j], ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(M[i, j]) > 15 else INK)
        ax.axhline(len(rows) - 1.5, color=SURF, lw=3)
        ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows] if ax is axes[0][0] else [])
        ax.set_title("T = %g epochs" % T)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Δ held-out MSE (%)")
    cb.set_ticks([-50, -10, -2, 0, 2, 10, 50])
    cb.set_ticklabels(["−50", "−10", "−2", "0", "+2", "+10", "+50"])
    fig.suptitle("Leave one out: % change in held-out MSE when one ingredient is "
                 "removed from e-means (orange = the ingredient helps, blue = it hurts)",
                 fontsize=10.5)
    fig.savefig(os.path.join(FIG, "leave_one_out.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(vals, datasets, epochs, ladder):
    fig, axes = grid(len(datasets))
    series = ladder[:5] + [KM]
    for ax, ds in zip(axes, datasets):
        ax.axhline(0, color=INK2, ls="--", lw=1)
        for i, v in enumerate(series):
            ys = [rel(vals, ds, T, v, LLOYD_CONVERGED, KM) for T in epochs]
            ys = [np.nan if y is None else y[0] for y in ys]
            ax.plot(epochs, ys, "-o", color=SERIES[i], lw=2, ms=4,
                    label=short(v).replace("\n", " ") if v != KM else "k-means (Lloyd)")
        ax.set_xscale("log")
        ax.set_xticks(epochs, ["%g" % T for T in epochs])
        ax.minorticks_off()
        ax.set_title(ds)
        ax.set_xlabel("epochs (passes over the data)")
        ax.set_ylabel("held-out MSE vs k-means@%d" % LLOYD_CONVERGED)
        pct_axis(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8)
    fig.suptitle("Held-out MSE vs epoch budget for each rung of the ladder "
                 "(dashed = k-means after %d iterations)" % LLOYD_CONVERGED, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(FIG, "curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
