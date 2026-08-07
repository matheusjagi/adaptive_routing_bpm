import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd

from src.config import PROCESSED_DIR, DECISION_BRANCHES
from src.cost_table import build_cost_signal

pd.set_option("display.width", 220)


def main():
    decisions = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    signal = build_cost_signal(decisions, window=30)

    cost_cols = [f"known_cost_{b}" for b in DECISION_BRANCHES]

    print("Cobertura (nº de decisões com valor conhecido, de", len(signal), "totais):")
    print(signal[cost_cols].notna().sum())
    print()

    print("Primeira decisão com TODOS os ramos já conhecidos:")
    fully_known = signal[signal[cost_cols].notna().all(axis=1)]
    print(fully_known["decision_time"].min(), f"-> instância nº {fully_known.index.min()} de {len(signal)}")
    print()

    print("Estatísticas dos sinais conhecidos (em horas), só onde disponível:")
    for col in cost_cols:
        vals_h = signal[col].dropna() / 3600
        print(f"  {col:35s} n={len(vals_h):6d}  média={vals_h.mean():10.2f}h  "
              f"desvio={vals_h.std():10.2f}h  min={vals_h.min():8.2f}h  max={vals_h.max():10.2f}h")
    print()

    print("Amostra cronológica (a cada ~3000 decisões) do custo conhecido de W_Validate application vs O_Accepted:")
    sample = fully_known.iloc[::3000]
    print(sample[["decision_time", "branch", "known_cost_W_Validate application", "known_cost_O_Accepted"]]
          .assign(**{
              "known_cost_W_Validate application": lambda d: d["known_cost_W_Validate application"] / 3600,
              "known_cost_O_Accepted": lambda d: d["known_cost_O_Accepted"] / 3600,
          }).to_string())

    out_path = os.path.join(PROCESSED_DIR, "a_validating_decisions_with_cost_signal.parquet")
    signal.to_parquet(out_path)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
