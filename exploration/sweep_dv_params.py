import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS
from src.cost_table import build_cost_signal
from src.evaluation import build_counterfactual_estimates, score_policy
from src.policies import hysteresis_policy

pd.set_option("display.width", 220)

WINDOWS = [30, 100, 300, 1000]
MARGINS = [0.0, 0.1, 0.2, 0.3]


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    cf = build_counterfactual_estimates(base)

    warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)

    rows = []
    for window in WINDOWS:
        signal = build_cost_signal(base, window=window)
        signal = signal.join(cf[[c for c in cf.columns if c.startswith("cf_cost_") or c == "oracle_choice"]])

        known_cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
        eval_mask = (
            (signal["decision_time"] >= warmup_end)
            & (~signal["in_buffer"])
            & (signal["branch"].isin(ROUTABLE_BRANCHES))
            & signal[known_cols].notna().all(axis=1)
        )
        eval_df = signal[eval_mask]

        for margin in MARGINS:
            choices = hysteresis_policy(signal, margin)
            score = score_policy(eval_df, choices.loc[eval_df.index], f"DV w={window} m={margin}")
            switches = (choices.loc[eval_df.index] != choices.loc[eval_df.index].shift(1)).sum()
            score["route_switches"] = int(switches)
            rows.append(score)

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nReferências: Estática/Centralizada regret=1.84h, optimal_rate=0.78 | Oráculo regret=0")


if __name__ == "__main__":
    main()
