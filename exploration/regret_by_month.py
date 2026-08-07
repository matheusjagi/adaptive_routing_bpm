import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS
from src.cost_table import build_cost_signal
from src.evaluation import build_counterfactual_estimates
from exploration.sweep_dv_params import hysteresis_policy

pd.set_option("display.width", 220)

DV_WINDOW = 300
DV_MARGIN = 0.1


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    cf = build_counterfactual_estimates(base)
    signal = build_cost_signal(base, window=DV_WINDOW)
    signal = signal.join(cf[[c for c in cf.columns if c.startswith("cf_cost_") or c == "oracle_choice"]])

    warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
    known_cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
    eval_mask = (
        (signal["decision_time"] >= warmup_end)
        & (~signal["in_buffer"])
        & (signal["branch"].isin(ROUTABLE_BRANCHES))
        & signal[known_cols].notna().all(axis=1)
    )
    df = signal[eval_mask].copy()

    df["dv_choice"] = hysteresis_policy(signal, DV_MARGIN).loc[df.index]
    df["static_choice"] = "O_Returned"

    cf_mat = df[[f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]].to_numpy()
    idx = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}
    oracle_cost = np.nanmin(cf_mat, axis=1)

    for pol in ["static_choice", "dv_choice"]:
        chosen = np.take_along_axis(
            cf_mat, df[pol].map(idx).to_numpy()[:, None], axis=1
        ).squeeze(1)
        df[f"regret_{pol}"] = (chosen - oracle_cost) / 3600

    df["month"] = df["decision_time"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        n=("branch", "size"),
        oracle_nao_returned=("oracle_choice", lambda s: (s != "O_Returned").mean()),
        regret_estatica_h=("regret_static_choice", "mean"),
        regret_dv_h=("regret_dv_choice", "mean"),
    )
    monthly["vantagem_dv_h"] = monthly["regret_estatica_h"] - monthly["regret_dv_h"]
    print(monthly.round(2).to_string())

    print("\nAgregado nos meses em que o oráculo prefere outro ramo em >=40% das decisões:")
    shift_months = monthly[monthly["oracle_nao_returned"] >= 0.4].index
    in_shift = df["month"].isin(shift_months)
    for label, mask in [("Meses de regime alternativo", in_shift), ("Meses de regime O_Returned", ~in_shift)]:
        sub = df[mask]
        print(f"  {label}: n={len(sub)} | regret estática={sub['regret_static_choice'].mean():.2f}h "
              f"| regret DV={sub['regret_dv_choice'].mean():.2f}h")


if __name__ == "__main__":
    main()
