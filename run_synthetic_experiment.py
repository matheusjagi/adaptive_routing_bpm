"""Experimento E2: caracterização sistemática do roteamento adaptativo sob deriva
controlada (semissintético, calibrado no log real; oráculo verdadeiro).

Protocolo (espelha o artigo, sem circularidade):
1. TUNING: grade (W, m) escolhida em seeds de tuning, sobre um regime de deriva moderado
   (mistura dos tipos a período 90d). Congela a melhor (W*, m*).
2. TABELA: por tipo de deriva (a período representativo), regret médio das políticas em
   seeds de TESTE independentes, com IC 95% e teste pareado (estática vs cost-tracking).
3. DIAGRAMA DE FASE: varre a taxa de deriva (período de 240d a 15d) e reporta o regret da
   estática vs cost-tracking -> mostra a banda onde o adaptativo compensa (o "seguro").
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from src.synthetic_drift import DriftEnv, DAY
from src.online_policies import run_policies
from src.config import RESULTS_DIR

HORIZON = 365 * DAY
WARMUP = 60 * DAY
ANN_RATE = 6.0                      # casos/h por ramo (W=1000 ~ 7 dias)
DECISION_STEP = 6.0                # decisão a cada 6h (4/dia)
AMP = 100.0
DRIFTS = ["stationary", "abrupt", "gradual", "oscillating"]

TUNE_SEEDS = list(range(0, 6))
TEST_SEEDS = list(range(100, 130))
GRID_W = [250, 500, 1000, 2000]
GRID_M = [0.0, 0.05, 0.1, 0.2]

DECISIONS = np.arange(WARMUP, HORIZON, DECISION_STEP)


def mean_regret(drift, period_h, seed, W, m):
    env = DriftEnv(drift, HORIZON, amp=AMP, period_h=period_h,
                   ann_rate_per_h=ANN_RATE, seed=seed)
    reg = run_policies(env, DECISIONS, warmup_h=WARMUP, W=W, m=m, use_tu=True)
    return {p: float(np.mean(v)) for p, v in reg.items()}


def tune():
    """Escolhe (W, m) minimizando o regret médio da cost-tracking sobre os tipos de deriva
    não estacionários a 90d, em seeds de tuning."""
    print("== TUNING (W, m) ==")
    best, best_cfg = np.inf, None
    for W in GRID_W:
        for m in GRID_M:
            regs = []
            for drift in ["abrupt", "gradual", "oscillating"]:
                for s in TUNE_SEEDS:
                    regs.append(mean_regret(drift, 90 * DAY, s, W, m)["cost_tracking"])
            mr = float(np.mean(regs))
            if mr < best:
                best, best_cfg = mr, (W, m)
    print(f"  melhor: W={best_cfg[0]}, m={best_cfg[1]}  (regret médio de tuning {best:.1f}h)\n")
    return best_cfg


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def table(Wf, mf):
    print("== TABELA por tipo de deriva (seeds de teste) ==")
    rows = []
    for drift in DRIFTS:
        per = {p: [] for p in ["static", "centralized", "cost_tracking"]}
        for s in TEST_SEEDS:
            r = mean_regret(drift, 90 * DAY, s, Wf, mf)
            for p in per:
                per[p].append(r[p])
        st, ce, ct = np.array(per["static"]), np.array(per["centralized"]), np.array(per["cost_tracking"])
        # teste pareado por seed: cost-tracking melhor que estática?
        diff = st - ct
        try:
            w_p = stats.wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            w_p = 1.0
        rows.append({
            "drift": drift,
            "static": st.mean(), "static_ci": ci95(st),
            "centralized": ce.mean(), "centralized_ci": ci95(ce),
            "cost_tracking": ct.mean(), "cost_tracking_ci": ci95(ct),
            "p_ct_better_than_static": w_p,
        })
        print(f"  {drift:12s} static={st.mean():6.1f}±{ci95(st):.1f}  "
              f"central={ce.mean():6.1f}±{ci95(ce):.1f}  "
              f"cost_tr={ct.mean():6.1f}±{ci95(ct):.1f}  "
              f"p(ct<static)={w_p:.4f}")
    print()
    return pd.DataFrame(rows)


def phase(Wf, mf):
    print("== DIAGRAMA DE FASE (varredura de período) ==")
    periods_d = [240, 180, 120, 90, 60, 45, 30, 20, 15]
    rows = []
    for per_d in periods_d:
        for drift in ["abrupt", "gradual", "oscillating"]:
            st, ct = [], []
            for s in TEST_SEEDS[:15]:
                r = mean_regret(drift, per_d * DAY, s, Wf, mf)
                st.append(r["static"]); ct.append(r["cost_tracking"])
            rows.append({"period_d": per_d, "drift": drift,
                         "static": float(np.mean(st)), "cost_tracking": float(np.mean(ct)),
                         "advantage": float(np.mean(st)) - float(np.mean(ct))})
        agg = [x for x in rows if x["period_d"] == per_d]
        adv = np.mean([x["advantage"] for x in agg])
        print(f"  período={per_d:4d}d  vantagem média (static-cost_tr)={adv:+6.1f}h")
    return pd.DataFrame(rows)


def main():
    Wf, mf = tune()
    out = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(out, exist_ok=True)

    tab = table(Wf, mf)
    tab.to_csv(os.path.join(out, "synthetic_table.csv"), index=False)

    ph = phase(Wf, mf)
    ph.to_csv(os.path.join(out, "synthetic_phase.csv"), index=False)

    with open(os.path.join(out, "synthetic_config.txt"), "w") as f:
        f.write(f"W={Wf}\nm={mf}\nann_rate={ANN_RATE}\namp={AMP}\nhorizon_d={HORIZON/DAY}\n"
                f"warmup_d={WARMUP/DAY}\ntest_seeds={len(TEST_SEEDS)}\n")
    print(f"\nConfig congelada: W={Wf}, m={mf}. Resultados em {out}/synthetic_*.csv")


if __name__ == "__main__":
    main()
