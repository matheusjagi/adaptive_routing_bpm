"""Escolhas mensais no teste (configuração congelada): fração de decisões em que cada política
(recomputação periódica, cost-tracking) e o oráculo-proxy escolhem cada stream, e as datas de
recomputação do baseline. Confirma a frase do §6: o baseline manteve A_Incomplete em outubro."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, TUNE_TEST_SPLIT, TU_DELTA_DAYS, TU_BETA, WARMUP_DAYS, RETRAIN_DAYS
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import centralized_policy, hysteresis_policy
from src.evaluation import build_counterfactual_estimates

CF = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]; KN = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet")).sort_values("decision_time").reset_index(drop=True)
cf = build_counterfactual_estimates(base)
s = combine_announcements(build_open_case_signal(build_cost_signal(base, window=1000), TU_DELTA_DAYS), TU_BETA).join(cf[CF + ["oracle_choice"]])
s["periodic"] = centralized_policy(s); s["cost_tracking"] = hysteresis_policy(s, 0.2)
T = s[(s["decision_time"] >= pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC")) & (~s["in_buffer"])
      & s["branch"].isin(ROUTABLE_BRANCHES) & s[KN].notna().all(axis=1)].copy()
T["month"] = T["decision_time"].dt.tz_localize(None).dt.to_period("M")
for p in ["periodic", "cost_tracking", "oracle_choice"]:
    print(f"\n{p}:"); print(T.groupby("month")[p].value_counts(normalize=True).round(2).unstack().fillna(0).to_string())
start = s["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
print("\nrecomputações do baseline:", [str((start + pd.Timedelta(days=RETRAIN_DAYS * k)).date()) for k in range(4, 11)])
