"""Gera o gráfico de regret mensal (teste, jul-dez/2016) para o artigo.
Saída: PDFs vetoriais (EN e PT) na pasta figures/ do artigo.

Codificação secundária (estilo de linha + marcador) além da cor, para leitura segura
em escala de cinza na impressão. Paleta validada (validate_palette.js): 2166AC/E08214/008B7D.
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "results", "metrics", "final_monthly_regret.csv")
FIG_DIR = "/home/matheusjagi/Documents/Mestrado/dissertação/artigo_adaptive_routing/figures"

BLUE = "#2166AC"   # cost-tracking (protagonista)
TEAL = "#008B7D"   # estática
ORANGE = "#E08214"  # centralizada
INK = "#333333"
GRID = "#D8D8D8"

MONTHS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load():
    static, central, cost = [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            static.append(float(row["Estática (BPMN fixo)"]))
            central.append(float(row["Centralizada (retreino 30d, janela 60d)"]))
            cost.append(float(row["Vetor de distância (congelado)"]))
    return static, central, cost


def make_plot(labels, out_path):
    static, central, cost = load()
    x = list(range(len(MONTHS)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.1))

    YMAX = 15.5
    # centralizada: clipar em YMAX (o pico de out fica fora da escala, anotado)
    central_clip = [min(v, YMAX) for v in central]

    ax.plot(x, central_clip, color=ORANGE, linestyle=":", marker="^", markersize=6.5,
            linewidth=1.8, label=labels["central"], zorder=2)
    ax.plot(x, static, color=TEAL, linestyle="--", marker="s", markersize=6,
            linewidth=1.8, label=labels["static"], zorder=3)
    ax.plot(x, cost, color=BLUE, linestyle="-", marker="o", markersize=6.5,
            linewidth=2.3, label=labels["cost"], zorder=4)

    # anotação do pico fora da escala (centralizada, outubro = 53.3h)
    ax.annotate(f"{labels['central_short']}: 53.3 h",
                xy=(3, YMAX), xytext=(3.05, YMAX - 2.1),
                fontsize=8.2, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))

    ax.set_ylim(-0.6, YMAX + 0.3)
    ax.set_xlim(-0.3, len(MONTHS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([labels["months"][i] for i in x])
    ax.set_ylabel(labels["ylabel"])

    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper center", frameon=False, fontsize=8.8, ncol=3,
              handlelength=2.4, columnspacing=1.3, bbox_to_anchor=(0.5, 1.13))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out_path)


def main():
    make_plot(
        labels={
            "static": "Static", "central": "Centralized", "cost": "Cost-tracking",
            "central_short": "Centralized", "ylabel": "Mean regret (h)",
            "months": MONTHS,
        },
        out_path=os.path.join(FIG_DIR, "monthly_regret.pdf"),
    )
    make_plot(
        labels={
            "static": "Estática", "central": "Centralizada", "cost": "Rastreamento de custo",
            "central_short": "Centralizada", "ylabel": "Regret médio (h)",
            "months": ["Jul", "Ago", "Set", "Out", "Nov", "Dez"],
        },
        out_path=os.path.join(FIG_DIR, "monthly_regret_pt.pdf"),
    )


if __name__ == "__main__":
    main()
