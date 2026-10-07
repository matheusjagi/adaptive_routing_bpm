"""Teste de independência: ramo roteável seguido em A_Validating x atributos do caso,
recurso e estado do processo. Qui-quadrado de Pearson + V de Cramér por variável.
Contínuos (RequestedAmount) em quartis; recursos raros (<200 decisões) agrupados em 'other'.
Universo: decisões avaliáveis (fora do buffer, ramo roteável, pós warm-up), como em
run_final_analysis.py. Saída: tabela + CSV em data/results/metrics/branch_independence.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from src.load_log import load_log
from src.config import PROCESSED_DIR, RESULTS_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS, DECISION_ACTIVITY

pd.set_option("display.width", 220)

def cramers_v(ct):
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return chi2, dof, p, v

dec = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
warmup_end = dec["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
dec = dec[(~dec["in_buffer"]) & dec["branch"].isin(ROUTABLE_BRANCHES) & (dec["decision_time"] >= warmup_end)].copy()

# recurso que executou o A_Validating e índice da rodada de validação no caso
log = load_log()
av = log[(log["concept:name"] == DECISION_ACTIVITY) & (log["lifecycle:transition"] == "complete")]
av = av[["case:concept:name", "time:timestamp", "org:resource"]].sort_values(["case:concept:name", "time:timestamp"])
av["validation_round"] = av.groupby("case:concept:name").cumcount() + 1
dec = dec.merge(av, left_on=["case_id", "decision_time"], right_on=["case:concept:name", "time:timestamp"], how="left")

dec["RequestedAmount_q"] = pd.qcut(dec["case:RequestedAmount"].astype(float), 4, duplicates="drop").astype(str)
top = dec["org:resource"].value_counts()
dec["resource_grp"] = dec["org:resource"].where(dec["org:resource"].map(top) >= 200, "other")
dec["round_grp"] = dec["validation_round"].clip(upper=4).astype(str).replace({"4": "4+"})
dec["month"] = dec["decision_time"].dt.to_period("M").astype(str)

rows = []
for var in ["case:LoanGoal", "case:ApplicationType", "RequestedAmount_q", "resource_grp", "round_grp", "month"]:
    ct = pd.crosstab(dec[var].fillna("NA"), dec["branch"])
    chi2, dof, p, v = cramers_v(ct)
    rows.append({"variable": var, "levels": ct.shape[0], "n": int(ct.to_numpy().sum()),
                 "chi2": chi2, "dof": dof, "p": p, "cramers_v": v})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

print("\nDistribuição do ramo por rodada de validação (linhas somam 1):")
print(pd.crosstab(dec["round_grp"], dec["branch"], normalize="index").round(3).to_string())
print("\nDistribuição do ramo por ApplicationType:")
print(pd.crosstab(dec["case:ApplicationType"], dec["branch"], normalize="index").round(3).to_string())

out = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(out, exist_ok=True)
res.to_csv(os.path.join(out, "branch_independence.csv"), index=False)
print(f"\nSalvo em {out}/branch_independence.csv")
