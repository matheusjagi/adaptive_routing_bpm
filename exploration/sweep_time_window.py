import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS
from src.cost_table import build_cost_signal_time
from src.evaluation import build_counterfactual_estimates, score_policy
from exploration.sweep_dv_params import hysteresis_policy

pd.set_option("display.width", 220)


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    cf = build_counterfactual_estimates(base)
    cf_keep = cf[[c for c in cf.columns if c.startswith("cf_cost_") or c == "oracle_choice"]]
    warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
    known_cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]

    rows = []
    monthly_best = None

    for days in [7, 14, 21, 30]:
        signal = build_cost_signal_time(base, window_days=days).join(cf_keep)
        eval_mask = (
            (signal["decision_time"] >= warmup_end)
            & (~signal["in_buffer"])
            & (signal["branch"].isin(ROUTABLE_BRANCHES))
            & signal[known_cols].notna().all(axis=1)
        )
        eval_df = signal[eval_mask]

        for margin in [0.0, 0.1, 0.2, 0.3]:
            choices = hysteresis_policy(signal, margin)
            s = score_policy(eval_df, choices.loc[eval_df.index], f"DV tempo D={days}d m={margin}")
            s["switches"] = int((choices.loc[eval_df.index] != choices.loc[eval_df.index].shift(1)).sum())
            rows.append(s)
            if monthly_best is None or s["mean_regret_h"] < monthly_best[0]["mean_regret_h"]:
                monthly_best = (s, eval_df, choices)

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nReferências: Estática 1.84h | DV contagem w=300 m=0.1: 3.87h | DV+TU: 3.71h")

    s_best, eval_df, choices = monthly_best
    print(f"\n=== Recorte mensal da melhor config ({s_best['policy']}) vs Estática ===")
    cf_cols = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]
    idx = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}
    cf_mat = eval_df[cf_cols].to_numpy()
    oracle = np.nanmin(cf_mat, axis=1)

    df = eval_df.copy()
    for name, ch in [("DV_tempo", choices.loc[eval_df.index]), ("Estatica", pd.Series("O_Returned", index=eval_df.index))]:
        chosen = np.take_along_axis(cf_mat, ch.map(idx).to_numpy()[:, None], axis=1).squeeze(1)
        df[f"regret_{name}"] = (chosen - oracle) / 3600
    df["month"] = df["decision_time"].dt.to_period("M")
    print(df.groupby("month")[["regret_Estatica", "regret_DV_tempo"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
