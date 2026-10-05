"""Plot simulation results (sim/results.csv -> docs/fig_*.pdf)."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {  # categorical slots 1-4 of the reference palette, fixed order
    "metaBRR": "#2a78d6",
    "BRR (pooled)": "#eb6834",
    "BRR (per cohort)": "#1baf7a",
    "metaBRR (shared β0)": "#eda100",
}
MARKERS = {"metaBRR": "o", "BRR (pooled)": "s", "BRR (per cohort)": "^",
           "metaBRR (shared β0)": "D"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def summarise(df, col):
    g = df.groupby(["h2", "rho", "method"])[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["het"] = 1 - out["rho"]
    return out


def accuracy_panels(df, col, ylabel, fname):
    s = summarise(df, col)
    h2s = sorted(s.h2.unique())
    fig, axes = plt.subplots(1, len(h2s), figsize=(1.9 * len(h2s) + 0.6, 2.6), sharey=False)
    for ax, h2 in zip(axes, h2s):
        for m in COLORS:
            d = s[(s.h2 == h2) & (s.method == m)].sort_values("het")
            ax.errorbar(d.het, d["mean"], yerr=d.se, color=COLORS[m], marker=MARKERS[m],
                        ms=4, lw=1.6, capsize=0, elinewidth=1, label=m)
        ax.set_title(f"$h^2$ = {h2}", fontsize=9)
        ax.set_xlabel("heterogeneity $1-\\rho$")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def gain_panel(df, fname):
    w = df.pivot_table(index=["seed", "h2", "rho"], columns="method", values="r2_g").reset_index()
    w["gain"] = w["metaBRR"] - w["BRR (pooled)"]
    seq = ["#9ec5f4", "#5a9be6", "#2a78d6", "#184f98"]  # one hue, light -> dark with h2
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for c, h2 in zip(seq, sorted(w.h2.unique())):
        d = w[w.h2 == h2].groupby("rho")["gain"].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(1 - d.rho, d["mean"], yerr=d["std"] / np.sqrt(d["count"]), color=c,
                    marker="o", ms=4, lw=1.6, elinewidth=1, label=f"$h^2$={h2}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("heterogeneity $1-\\rho$")
    ax.set_ylabel("$R^2_g$(metaBRR) $-$ $R^2_g$(BRR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def rho_panel(df, fname):
    d = df[df.method == "metaBRR"]
    seq = ["#9ec5f4", "#5a9be6", "#2a78d6", "#184f98"]
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    for i, (c, h2) in enumerate(zip(seq, sorted(d.h2.unique()))):
        dd = d[d.h2 == h2]
        jitter = (i - 1.5) * 0.012
        ax.scatter(dd.rho + jitter, dd.rho_hat, s=18, color=c, edgecolor="white",
                   linewidth=0.6, label=f"$h^2$={h2}", zorder=3)
    ax.set_xlabel("true cross-cohort correlation $\\rho$")
    ax.set_ylabel("estimated $\\hat\\rho = \\lambda_0^{-1}/(\\lambda_0^{-1}+\\lambda_\\delta^{-1})$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def table(df, fname):
    s = summarise(df, "r2_g")
    w = s.pivot_table(index=["h2", "rho"], columns="method", values="mean").reset_index()
    cols = ["BRR (pooled)", "BRR (per cohort)", "metaBRR (shared β0)", "metaBRR"]
    lines = []
    for _, r in w.iterrows():
        vals = [r[c] for c in cols]
        best = max(vals)
        cells = [f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}" for v in vals]
        lines.append(f"{r.h2:g} & {r.rho:g} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{rrcccc}\n\\toprule\n$h^2$ & $\\rho$ & BRR (pooled) & "
            "BRR (per cohort) & metaBRR (shared $\\hat{\\bm\\beta}_0$) & metaBRR \\\\\n\\midrule\n")
    open(fname, "w").write(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sim/results.csv"
    df = pd.read_csv(path)
    accuracy_panels(df, "r2_g", "test $R^2$ with genetic value", "docs/fig_r2g.pdf")
    accuracy_panels(df, "r2_y", "test $R^2$ (phenotype)", "docs/fig_r2y.pdf")
    gain_panel(df, "docs/fig_gain.pdf")
    rho_panel(df, "docs/fig_rho.pdf")
    table(df, "docs/table_r2g.tex")
    t = df.groupby("method").time.median()
    open("docs/timing.tex", "w").write(
        "\\newcommand{\\tmeta}{%.1f}\n\\newcommand{\\tbrr}{%.1f}\n"
        % (t["metaBRR"], t["BRR (pooled)"]))
