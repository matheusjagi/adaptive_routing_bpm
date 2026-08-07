import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from src.load_log import load_log

pd.set_option("display.width", 200)

CANDIDATES = [
    "A_Create Application",
    "A_Submitted",
    "A_Validating",
    "A_Incomplete",
    "A_Pending",
    "O_Created",
    "O_Returned",
    "O_Refused",
    "O_Sent (mail and online)",
    "O_Cancelled",
    "W_Validate application",
]


def main():
    df = load_log()
    complete = df[df["lifecycle:transition"] == "complete"].copy()
    complete = complete.sort_values(["case:concept:name", "time:timestamp"])
    case_end = complete.groupby("case:concept:name")["time:timestamp"].max().rename("case_end")
    complete["next_activity"] = complete.groupby("case:concept:name")["concept:name"].shift(-1)

    print(f"{'Atividade':30s} {'Ramo seguinte':30s} {'n':>6s} {'mediana(h)':>12s} {'p75(h)':>10s} {'max(h)':>10s}")
    print("-" * 100)

    for activity in CANDIDATES:
        sub = complete[complete["concept:name"] == activity].copy()
        sub = sub.merge(case_end, on="case:concept:name", how="left")
        sub["cost_h"] = (sub["case_end"] - sub["time:timestamp"]).dt.total_seconds() / 3600.0

        branch_counts = sub["next_activity"].value_counts()
        branches = branch_counts[branch_counts >= 200].index.tolist()
        if len(branches) < 2:
            continue

        for branch in branches:
            g = sub[sub["next_activity"] == branch]
            print(f"{activity:30s} {str(branch):30s} {len(g):6d} {g['cost_h'].median():12.4f} "
                  f"{g['cost_h'].quantile(0.75):10.4f} {g['cost_h'].max():10.4f}")
        print()


if __name__ == "__main__":
    main()
