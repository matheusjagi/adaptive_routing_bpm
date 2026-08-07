"""Figura do diagrama de fase do E2: regret vs. período de deriva (estática vs. cost-tracking),
média sobre os tipos de deriva não estacionários. Saída: PDFs (EN e PT) na pasta figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(_PROJECT_ROOT, "data", "results", "metrics", "synthetic_phase.csv")
FIG_DIR = "/home/matheusjagi/Documents/Mestrado/dissertação/artigo_adaptive_routing/figures"

TEAL = "#008B7D"   # estática
BLUE = "#2166AC"   # cost-tracking
INK = "#333333"
GRID = "#D8D8D8"
BAND = "#2166AC"


def make(labels, out_path):
    df = pd.read_csv(CSV)
    g = df.groupby("period_d")[["static", "cost_tracking"]].mean().sort_index()
    x = g.index.values
    static = g["static"].values
    cost = g["cost_tracking"].values

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9.5,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    # banda: onde o adaptativo vence (cost < static)
    ax.fill_between(x, cost, static, where=(cost < static), color=BAND, alpha=0.10,
                    interpolate=True, label=labels["band"])

    ax.plot(x, static, color=TEAL, linestyle="--", marker="s", markersize=6, linewidth=1.9,
            label=labels["static"], zorder=3)
    ax.plot(x, cost, color=BLUE, linestyle="-", marker="o", markersize=6.5, linewidth=2.3,
            label=labels["cost"], zorder=4)

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.invert_xaxis()  # deriva lenta (esq.) -> rápida (dir.)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_ylim(bottom=-2)

    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.8)

    # anotação do regime
    ax.annotate(labels["slow"], xy=(x.max(), 3), xytext=(x.max(), 3),
                fontsize=8, color=INK, ha="left", va="bottom")
    ax.annotate(labels["fast"], xy=(x.min(), 3), xytext=(x.min(), 3),
                fontsize=8, color=INK, ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out_path)


def main():
    make(
        labels={"static": "Static", "cost": "Cost-tracking", "band": "adaptive wins",
                "xlabel": "Drift period (days, log scale)", "ylabel": "Mean regret (h)",
                "slow": "slow drift", "fast": "fast drift"},
        out_path=os.path.join(FIG_DIR, "phase_diagram.pdf"),
    )
    make(
        labels={"static": "Estática", "cost": "Rastreamento de custo",
                "band": "adaptativo vence",
                "xlabel": "Período de deriva (dias, escala log)", "ylabel": "Regret médio (h)",
                "slow": "deriva lenta", "fast": "deriva rápida"},
        out_path=os.path.join(FIG_DIR, "phase_diagram_pt.pdf"),
    )


if __name__ == "__main__":
    main()
