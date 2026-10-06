"""Em qual período e configuração surgem flapping e lock-in reportados no artigo (§6)?
Reproduz sweep_dv_params.py (janela por contagem) e sweep_time_window.py (janela temporal),
separando o regret e as trocas por período: tuning, teste e pós-warm-up completo."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS, TUNE_TEST_SPLIT
from src.cost_table import build_cost_signal, build_cost_signal_time
from src.policies import hysteresis_policy
from src.evaluation import build_counterfactual_estimates, score_policy

KN = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet")).sort_values("decision_time").reset_index(drop=True)
cf = build_counterfactual_estimates(base)
cf_keep = cf[[c for c in cf.columns if c.startswith("cf_cost_")] + ["oracle_choice"]]
w_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
split = pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC"); t_max = base["decision_time"].max() + pd.Timedelta(days=1)
rows = []
cfgs = [("count", w, m) for w in [30, 100, 300, 1000] for m in [0.0, 0.1, 0.3]] + [("time", d, 0.3) for d in [7, 14, 21, 30]]
for kind, w, m in cfgs:
    s = (build_cost_signal(base, window=w) if kind == "count" else build_cost_signal_time(base, window_days=w)).join(cf_keep)
    ch = hysteresis_policy(s, m)
    for per, lo, hi in [("tuning", w_end, split), ("teste", split, t_max), ("pós-warm-up", w_end, t_max)]:
        d = s[(s["decision_time"] >= lo) & (s["decision_time"] < hi) & (~s["in_buffer"]) & s["branch"].isin(ROUTABLE_BRANCHES) & s[KN].notna().all(axis=1)]
        c = ch.loc[d.index]
        rows.append({"janela": f"{kind}:{w}", "m": m, "período": per, "regret_h": score_policy(d, c, "")["mean_regret_h"],
                     "trocas": int((c != c.shift(1)).sum())})
r = pd.DataFrame(rows).pivot_table(index=["janela", "m"], columns="período", values=["regret_h", "trocas"], aggfunc="first")
print(r.round(2).to_string())
