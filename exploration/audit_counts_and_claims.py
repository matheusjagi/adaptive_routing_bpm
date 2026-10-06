"""Auditoria de números do artigo a partir do pipeline canônico (sem alterar nada):
 1. Reconciliação de contagens (38.688 / 36.578 / tuning / teste).
 2. Ramo fixado pela política estática e frequência com que o oráculo diverge dele.
 3. Candidatos para a origem de '48%' e '101 h' (§5 do artigo).
 4. Decomposição mensal do regret (n por mês) e contribuição de cada mês ao gap DV x centralizada.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.config import PROCESSED_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS, TUNE_TEST_SPLIT, TU_DELTA_DAYS, TU_BETA
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import static_policy, centralized_policy, hysteresis_policy
from src.evaluation import build_counterfactual_estimates

pd.set_option("display.width", 220)
CF = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]
KN = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
IDX = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}

base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
base = base.sort_values("decision_time").reset_index(drop=True)
print("== 1. Contagens ==")
print(f"  decisões após censura camada 1: {len(base)} (+87 removidas = {len(base)+87})")
print(f"  fora do buffer (todas as 5 saídas): {(~base['in_buffer']).sum()}")
print(f"  fora do buffer e ramo roteável: {((~base['in_buffer']) & base['branch'].isin(ROUTABLE_BRANCHES)).sum()}")

cf = build_counterfactual_estimates(base)
warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
split = pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC")
sig = build_cost_signal(base, window=1000)
sig = combine_announcements(build_open_case_signal(sig, delta_days=TU_DELTA_DAYS), beta=TU_BETA)
sig = sig.join(cf[CF + ["oracle_choice"]])
ok = (~sig["in_buffer"]) & sig["branch"].isin(ROUTABLE_BRANCHES) & sig[KN].notna().all(axis=1)
tune = sig[ok & (sig["decision_time"] >= warmup_end) & (sig["decision_time"] < split)]
test = sig[ok & (sig["decision_time"] >= split)]
print(f"  avaliáveis pós warm-up: {len(tune)+len(test)} | tuning: {len(tune)} | teste: {len(test)}")
print(f"  warm-up termina em {warmup_end:%Y-%m-%d}; última decisão de teste {test['decision_time'].max():%Y-%m-%d}")

print("\n== 2. Política estática ==")
st = static_policy(sig)
print(f"  ramo fixado: {st.iloc[0]}")
for name, d in [("pós warm-up (tuning+teste)", pd.concat([tune, test])), ("teste", test)]:
    print(f"  {name}: oráculo escolhe o ramo estático em {(d['oracle_choice'] == st.iloc[0]).mean():.3f}")

print("\n== 3. Origem de '48%' e '101 h' (candidatos) ==")
cfm = sig[CF].to_numpy()
gap = (np.nanmax(cfm, axis=1) - np.nanmin(cfm, axis=1)) / 3600
allok = sig[CF].notna().all(axis=1)
glob_leader = base[base["branch"].isin(ROUTABLE_BRANCHES)].groupby("branch")["cost_seconds"].mean().idxmin()
print(f"  líder global (média de todo o log, roteáveis): {glob_leader}")
for label, m in [("todas as decisões c/ oráculo completo", allok), ("avaliáveis pós warm-up", allok & ok & (sig['decision_time'] >= warmup_end))]:
    print(f"  [{label}] n={int(m.sum())}: oráculo != líder global em {(sig.loc[m,'oracle_choice'] != glob_leader).mean():.3f} | gap médio max-min (oráculo) = {gap[m.to_numpy()].mean():.1f} h")
for w in [30, 300]:
    s = build_cost_signal(base, window=w)
    k = s[KN]
    m = k.notna().all(axis=1)
    lead = k[m].idxmin(axis=1).str.replace("known_cost_", "", regex=False)
    kg = (k[m].max(axis=1) - k[m].min(axis=1)) / 3600
    print(f"  [tabela trailing W={w}, todas as decisões] n={int(m.sum())}: líder corrente != líder global em {(lead != glob_leader).mean():.3f} | gap médio max-min = {kg.mean():.1f} h")

print("\n== 4. Decomposição mensal do teste ==")
pol = {"static": st, "central": centralized_policy(sig), "dv": hysteresis_policy(sig, 0.2)}
T = test.copy()
orc = np.nanmin(T[CF].to_numpy(), axis=1)
for p, ch in pol.items():
    c = ch.loc[T.index].map(IDX).to_numpy()
    T[p] = (np.take_along_axis(T[CF].to_numpy(), c[:, None], axis=1).squeeze(1) - orc) / 3600
T["month"] = T["decision_time"].dt.tz_localize(None).dt.to_period("M")
mm = T.groupby("month").agg(n=("dv", "size"), static=("static", "mean"), central=("central", "mean"), dv=("dv", "mean"))
mm["central_minus_dv_share_of_total_gap"] = ((T.groupby("month")["central"].sum() - T.groupby("month")["dv"].sum())
                                              / (T["central"].sum() - T["dv"].sum()))
print(mm.round(3).to_string())
print(f"  agregado: static={T['static'].mean():.2f} central={T['central'].mean():.2f} dv={T['dv'].mean():.2f}")
no_oct = T["month"] != pd.Period("2016-10")
print(f"  sem outubro: central={T.loc[no_oct,'central'].mean():.2f} dv={T.loc[no_oct,'dv'].mean():.2f}")
