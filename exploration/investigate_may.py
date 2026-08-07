import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd

from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES
from src.cost_table import build_cost_signal
from src.evaluation import build_counterfactual_estimates

pd.set_option("display.width", 220)


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    signal = build_cost_signal(base, window=300)
    cf = build_counterfactual_estimates(base)
    signal = signal.join(cf[[c for c in cf.columns if c.startswith("cf_cost_")]])

    may = signal[(signal["decision_time"] >= "2016-05-01") & (signal["decision_time"] < "2016-06-01")]

    print("=== MAIO/2016: o que o DV via (tabela) vs o que era verdade (contrafactual ±14d) ===")
    for b in ROUTABLE_BRANCHES:
        known = may[f"known_cost_{b}"].mean() / 3600
        cf_v = may[f"cf_cost_{b}"].mean() / 3600
        print(f"  {b:25s} tabela(trailing)={known:8.1f}h | contrafactual={cf_v:8.1f}h | erro={known-cf_v:+8.1f}h")

    print("\n=== Custo real por mês de ENTRADA vs mês de RESOLUÇÃO (viés de amostragem por resolução) ===")
    for b in ROUTABLE_BRANCHES:
        taken = signal[signal["branch"] == b].copy()
        taken["entry_month"] = taken["decision_time"].dt.to_period("M")
        taken["end_month"] = taken["case_end"].dt.to_period("M")
        months = [pd.Period(m) for m in ["2016-03", "2016-04", "2016-05", "2016-06"]]
        ent = taken[taken["entry_month"].isin(months)].groupby("entry_month")["cost_seconds"].agg(["count", "mean"])
        res = taken[taken["end_month"].isin(months)].groupby("end_month")["cost_seconds"].agg(["count", "mean"])
        ent["mean"] = ent["mean"] / 3600
        res["mean"] = res["mean"] / 3600
        tbl = ent.join(res, lsuffix="_entrada", rsuffix="_resolucao")
        print(f"\n{b} (custo médio em horas):")
        print(tbl.round(1).to_string())


if __name__ == "__main__":
    main()
