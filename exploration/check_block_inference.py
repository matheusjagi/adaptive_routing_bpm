"""Robustez da inferência à dependência temporal: os testes do artigo tratam as 16.983
decisões de teste como independentes. Aqui: (a) Wilcoxon pareado sobre médias SEMANAIS de
regret; (b) IC 95% por bootstrap de blocos semanais (2000 reamostragens) da diferença média
de regret rival - DV. Mesmas políticas e configuração congelada de run_final_analysis.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, TUNE_TEST_SPLIT, TU_DELTA_DAYS, TU_BETA
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import static_policy, centralized_policy, hysteresis_policy
from src.evaluation import build_counterfactual_estimates

CF = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]; KN = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
IDX = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}
base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet")).sort_values("decision_time").reset_index(drop=True)
cf = build_counterfactual_estimates(base)
s = combine_announcements(build_open_case_signal(build_cost_signal(base, window=1000), delta_days=TU_DELTA_DAYS), beta=TU_BETA).join(cf[CF + ["oracle_choice"]])
T = s[(s["decision_time"] >= pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC")) & (~s["in_buffer"]) & s["branch"].isin(ROUTABLE_BRANCHES) & s[KN].notna().all(axis=1)].copy()
orc = np.nanmin(T[CF].to_numpy(), axis=1)
pol = {"static": static_policy(s), "central": centralized_policy(s), "dv": hysteresis_policy(s, 0.2)}
for p, ch in pol.items():
    T[p] = (np.take_along_axis(T[CF].to_numpy(), ch.loc[T.index].map(IDX).to_numpy()[:, None], axis=1).squeeze(1) - orc) / 3600
T["log"] = (np.take_along_axis(T[CF].to_numpy(), T["branch"].map(IDX).to_numpy()[:, None], axis=1).squeeze(1) - orc) / 3600
T["week"] = T["decision_time"].dt.tz_localize(None).dt.to_period("W")
wk = T.groupby("week")[["static", "central", "dv", "log"]].mean()
rng = np.random.default_rng(42); weeks = wk.index.to_numpy(); n = len(weeks)
grp = {w: g for w, g in T.groupby("week")}
print(f"semanas no teste: {n}")
for rival in ["central", "log", "static"]:
    d = (wk[rival] - wk["dv"]).to_numpy()
    p_greater = stats.wilcoxon(d, alternative="greater").pvalue
    p_less = stats.wilcoxon(d, alternative="less").pvalue
    boots = []
    for _ in range(2000):
        pick = rng.choice(weeks, size=n, replace=True)
        bb = pd.concat([grp[w] for w in pick])
        boots.append(bb[rival].mean() - bb["dv"].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{rival:8s} - dv: diff média por decisão={T[rival].mean()-T['dv'].mean():+.2f} h | IC95 bootstrap semanal=[{lo:+.2f}, {hi:+.2f}] | "
          f"Wilcoxon semanal p(rival pior)={p_greater:.3f} p(rival melhor)={p_less:.3f} | semanas rival>dv: {(d>0).sum()}, rival<dv: {(d<0).sum()}, empate: {(d==0).sum()}")
