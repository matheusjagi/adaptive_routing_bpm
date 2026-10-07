"""Semântica dos sucessores de A_Validating no log bruto (todos os lifecycles).
Para cada A_Validating 'complete' com sucessor (entre os 'complete') roteável:
 (a) intervalo de tempo até o evento-ramo;
 (b) o evento imediatamente anterior e posterior (qualquer lifecycle);
 (c) se W_Validate application estava ABERTO (start/resume sem suspend/complete/abort
     posterior) no instante de A_Validating -> então o 'ramo' W_Validate application
     é o fechamento do MESMO work item;
 (d) o que vem depois do evento-ramo (próximo 'complete').
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.load_log import load_log
from src.config import DECISION_ACTIVITY, ROUTABLE_BRANCHES

pd.set_option("display.width", 220)

df = load_log().sort_values(["case:concept:name", "time:timestamp"], kind="stable").reset_index(drop=True)
df["label"] = df["concept:name"] + "|" + df["lifecycle:transition"]
g = df.groupby("case:concept:name", sort=False)
df["prev_label"] = g["label"].shift(1)
df["next_label"] = g["label"].shift(-1)

comp = df[df["lifecycle:transition"] == "complete"].copy()
cg = comp.groupby("case:concept:name", sort=False)
comp["next_c"] = cg["concept:name"].shift(-1)
comp["next_c_time"] = cg["time:timestamp"].shift(-1)
comp["next2_c"] = cg["concept:name"].shift(-2)
dec = comp[(comp["concept:name"] == DECISION_ACTIVITY) & comp["next_c"].isin(ROUTABLE_BRANCHES)].copy()
dec["gap_s"] = (dec["next_c_time"] - dec["time:timestamp"]).dt.total_seconds()

print("(a) Intervalo A_Validating -> evento-ramo (segundos):")
print(dec.groupby("next_c")["gap_s"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
print("\n    Fração com intervalo < 1 s / < 60 s:")
print(dec.groupby("next_c")["gap_s"].agg(lt1s=lambda s: (s < 1).mean(), lt60s=lambda s: (s < 60).mean()).round(3).to_string())

print("\n(b) Evento imediatamente ANTERIOR a A_Validating (qualquer lifecycle), por ramo:")
for b in ROUTABLE_BRANCHES:
    vc = df.loc[dec[dec["next_c"] == b].index, "prev_label"].value_counts(normalize=True).head(4).round(3)
    print(f"  {b}: {vc.to_dict()}")
print("\n    Evento imediatamente POSTERIOR a A_Validating (qualquer lifecycle), por ramo:")
for b in ROUTABLE_BRANCHES:
    vc = df.loc[dec[dec["next_c"] == b].index, "next_label"].value_counts(normalize=True).head(4).round(3)
    print(f"  {b}: {vc.to_dict()}")

# (c) estado do work item W_Validate application no instante de A_Validating
wv = df[df["concept:name"] == "W_Validate application"][["case:concept:name", "time:timestamp", "lifecycle:transition"]]
wv = wv.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
open_state = {"start": True, "resume": True, "schedule": False, "suspend": False,
              "complete": False, "ate_abort": False, "withdraw": False}
wv["is_open_after"] = wv["lifecycle:transition"].map(open_state)
wv = wv.rename(columns={"time:timestamp": "t_wv"})
d2 = dec[["case:concept:name", "time:timestamp", "next_c"]].reset_index().rename(columns={"index": "row"})
d2 = d2.sort_values("time:timestamp")
wv = wv.sort_values("t_wv")
m = pd.merge_asof(d2, wv, left_on="time:timestamp", right_on="t_wv",
                  by="case:concept:name", direction="backward", allow_exact_matches=True)
m["wv_open_at_decision"] = m["is_open_after"].fillna(False).astype(bool)
print("\n(c) W_Validate application ABERTO (último lifecycle = start/resume) no instante de A_Validating:")
print(m.groupby("next_c")["wv_open_at_decision"].mean().round(3).to_string())
print("    Último lifecycle de W_Validate application antes de A_Validating, por ramo:")
print(m.groupby("next_c")["lifecycle:transition"].value_counts(normalize=True).round(3).unstack().fillna(0).to_string())

print("\n(d) Próximo 'complete' DEPOIS do evento-ramo (top 5), por ramo:")
for b in ROUTABLE_BRANCHES:
    vc = dec[dec["next_c"] == b]["next2_c"].value_counts(normalize=True).head(5).round(3)
    print(f"  {b}: {vc.to_dict()}")
