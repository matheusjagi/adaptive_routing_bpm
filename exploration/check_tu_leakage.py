"""Vazamento dos hiperparâmetros de triggered updates (Delta, beta).
O artigo/config usam Delta=14 d e beta=1.0, escolhidos em sweep_triggered_updates.py sobre
TODO o período pós warm-up (que inclui o período de teste). Aqui a mesma grade é avaliada
(a) só no período de tuning e (b) só no teste, com a configuração de exploração (W=300, m=0.1)
e com a configuração congelada do artigo (W=1000, m=0.2). Não altera nenhum resultado.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS, TUNE_TEST_SPLIT
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import hysteresis_policy
from src.evaluation import build_counterfactual_estimates, score_policy

KN = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet")).sort_values("decision_time").reset_index(drop=True)
cf = build_counterfactual_estimates(base)
cf_keep = cf[[c for c in cf.columns if c.startswith("cf_cost_")] + ["oracle_choice"]]
w_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
split = pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC")

def mask(s, lo, hi):
    return ((s["decision_time"] >= lo) & (s["decision_time"] < hi) & (~s["in_buffer"])
            & s["branch"].isin(ROUTABLE_BRANCHES) & s[KN].notna().all(axis=1))

rows = []
t_max = base["decision_time"].max() + pd.Timedelta(days=1)
for W, m in [(300, 0.1), (1000, 0.2)]:
    resolved = build_cost_signal(base, window=W)
    for delta in [7, 14]:
        with_open = build_open_case_signal(resolved, delta_days=delta)
        for beta in [1.0, 1.5, 2.0]:
            s = combine_announcements(with_open, beta=beta).join(cf_keep)
            ch = hysteresis_policy(s, m)
            for per, lo, hi in [("tuning", w_end, split), ("teste", split, t_max), ("pós-warm-up", w_end, t_max)]:
                d = s[mask(s, lo, hi)]
                r = score_policy(d, ch.loc[d.index], "")
                rows.append({"W": W, "m": m, "delta": delta, "beta": beta, "period": per, "regret_h": r["mean_regret_h"]})
    s = resolved.join(cf_keep); ch = hysteresis_policy(s, m)
    for per, lo, hi in [("tuning", w_end, split), ("teste", split, t_max), ("pós-warm-up", w_end, t_max)]:
        d = s[mask(s, lo, hi)]
        rows.append({"W": W, "m": m, "delta": "off", "beta": "off", "period": per, "regret_h": score_policy(d, ch.loc[d.index], "")["mean_regret_h"]})

r = pd.DataFrame(rows).pivot_table(index=["W", "m", "delta", "beta"], columns="period", values="regret_h", aggfunc="first")
print(r[["tuning", "teste", "pós-warm-up"]].round(2).to_string())
for (W, m), g in r.groupby(level=[0, 1]):
    g2 = g.drop(index="off", level=2, errors="ignore")
    print(f"W={W} m={m}: melhor (Delta,beta) no tuning = {g2['tuning'].idxmin()[2:]} | no pós-warm-up = {g2['pós-warm-up'].idxmin()[2:]}")
