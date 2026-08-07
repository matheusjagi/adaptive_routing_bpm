import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd

from src.config import PROCESSED_DIR
from src.load_log import load_log

pd.set_option("display.width", 220)

TERMINAL_STATES = ["A_Pending", "A_Cancelled", "A_Denied"]


def main():
    df = load_log()
    log_end = df["time:timestamp"].max()
    print(f"Fim do log: {log_end}")

    complete = df[df["lifecycle:transition"] == "complete"]
    terminal_cases = set(complete[complete["concept:name"].isin(TERMINAL_STATES)]["case:concept:name"])
    all_cases = set(df["case:concept:name"])
    open_cases = all_cases - terminal_cases
    print(f"\nCases totais: {len(all_cases)} | com estado terminal (A_Pending/A_Cancelled/A_Denied): "
          f"{len(terminal_cases)} | SEM estado terminal (abertos no fim do log): {len(open_cases)}")

    last_event = df.groupby("case:concept:name")["time:timestamp"].max()
    open_last = last_event[last_event.index.isin(open_cases)]
    print("\nÚltimo evento dos cases abertos (deveriam se concentrar perto do fim do log):")
    print(open_last.dt.to_period("M").value_counts().sort_index())

    decisions = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    decisions["month"] = decisions["decision_time"].dt.to_period("M")
    decisions["censored_case"] = ~decisions["case_id"].isin(terminal_cases)

    print(f"\nDecisões A_Validating em cases abertos (custo truncado com certeza): "
          f"{decisions['censored_case'].sum()} de {len(decisions)}")
    print(decisions[decisions["censored_case"]]["month"].value_counts().sort_index())

    print("\nCusto observado (mediana, em horas) por mês da decisão — queda no fim indica truncamento:")
    monthly = decisions.groupby("month")["cost_seconds"].agg(["count", "median", lambda s: s.quantile(0.95)])
    monthly.columns = ["n", "mediana_s", "p95_s"]
    monthly["mediana_h"] = monthly["mediana_s"] / 3600
    monthly["p95_dias"] = monthly["p95_s"] / 86400
    print(monthly[["n", "mediana_h", "p95_dias"]].to_string())

    print("\nPercentis do custo (todas as decisões, em dias) — base para o buffer:")
    for q in [0.90, 0.95, 0.99]:
        print(f"  p{int(q*100)}: {decisions['cost_seconds'].quantile(q)/86400:8.1f} dias")


if __name__ == "__main__":
    main()
