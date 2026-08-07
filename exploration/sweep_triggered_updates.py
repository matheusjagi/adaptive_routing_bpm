import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import centralized_policy
from src.evaluation import build_counterfactual_estimates, score_policy
from exploration.sweep_dv_params import hysteresis_policy

pd.set_option("display.width", 220)

DV_WINDOW = 300
DV_MARGIN = 0.1
DELTAS = [7, 14]
BETAS = [1.0, 1.5, 2.0]


def eval_mask_for(signal, warmup_end):
    known_cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
    return (
        (signal["decision_time"] >= warmup_end)
        & (~signal["in_buffer"])
        & (signal["branch"].isin(ROUTABLE_BRANCHES))
        & signal[known_cols].notna().all(axis=1)
    )


def monthly_breakdown(eval_df, choices, cf_mat, oracle_cost, idx, label):
    df = eval_df.copy()
    df["choice"] = choices
    chosen = np.take_along_axis(cf_mat, df["choice"].map(idx).to_numpy()[:, None], axis=1).squeeze(1)
    df["regret_h"] = (chosen - oracle_cost) / 3600
    df["month"] = df["decision_time"].dt.to_period("M")
    return df.groupby("month")["regret_h"].mean().rename(label)


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    cf = build_counterfactual_estimates(base)
    cf_keep = cf[[c for c in cf.columns if c.startswith("cf_cost_") or c == "oracle_choice"]]
    warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)

    resolved_signal = build_cost_signal(base, window=DV_WINDOW)

    rows = []
    best = None

    for delta in DELTAS:
        with_open = build_open_case_signal(resolved_signal, delta_days=delta)
        for beta in BETAS:
            signal = combine_announcements(with_open, beta=beta).join(cf_keep)
            eval_df = signal[eval_mask_for(signal, warmup_end)]
            choices = hysteresis_policy(signal, DV_MARGIN)
            s = score_policy(eval_df, choices.loc[eval_df.index], f"DV+TU d={delta} b={beta}")
            s["switches"] = int((choices.loc[eval_df.index] != choices.loc[eval_df.index].shift(1)).sum())
            rows.append(s)
            if best is None or s["mean_regret_h"] < best[0]["mean_regret_h"]:
                best = (s, signal, choices)

    # referências na mesma base de avaliação
    ref_signal = resolved_signal.join(cf_keep)
    eval_df = ref_signal[eval_mask_for(ref_signal, warmup_end)]

    dv_choices = hysteresis_policy(ref_signal, DV_MARGIN)
    rows.append(score_policy(eval_df, dv_choices.loc[eval_df.index], "DV sem TU (w=300 m=0.1)"))

    static_choices = pd.Series("O_Returned", index=eval_df.index)
    rows.append(score_policy(eval_df, static_choices, "Estática"))

    central_choices = centralized_policy(ref_signal, window_days=60)
    rows.append(score_policy(eval_df, central_choices.loc[eval_df.index].dropna(), "Centralizada (janela 60d)"))

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # recorte mensal da melhor configuração vs estática
    print("\n=== Recorte mensal (regret médio, h) — melhor DV+TU vs Estática vs DV sem TU ===")
    s_best, best_signal, best_choices = best
    print(f"Melhor configuração: {s_best['policy']}")

    best_eval = best_signal[eval_mask_for(best_signal, warmup_end)]
    cf_cols = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]
    idx = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}

    cf_mat_best = best_eval[cf_cols].to_numpy()
    oracle_best = np.nanmin(cf_mat_best, axis=1)
    m1 = monthly_breakdown(best_eval, best_choices.loc[best_eval.index], cf_mat_best, oracle_best, idx, "DV+TU")

    cf_mat_ref = eval_df[cf_cols].to_numpy()
    oracle_ref = np.nanmin(cf_mat_ref, axis=1)
    m2 = monthly_breakdown(eval_df, static_choices, cf_mat_ref, oracle_ref, idx, "Estática")
    m3 = monthly_breakdown(eval_df, dv_choices.loc[eval_df.index], cf_mat_ref, oracle_ref, idx, "DV sem TU")

    print(pd.concat([m2, m3, m1], axis=1).round(2).to_string())


if __name__ == "__main__":
    main()
