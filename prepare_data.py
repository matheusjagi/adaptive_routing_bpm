import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_log import load_log
from src.decision_dataset import build_decision_dataset, apply_censoring
from src.config import PROCESSED_DIR


def main():
    print("Carregando log (gera cache parquet na primeira execução)...")
    df = load_log()
    print(f"Eventos: {len(df)} | Cases: {df['case:concept:name'].nunique()}")

    decisions = build_decision_dataset(df)
    print(f"\nInstâncias de decisão (A_Validating com ramo conhecido): {len(decisions)}")

    decisions, cens_info = apply_censoring(decisions, df)
    print(f"Censura: {cens_info['n_open_dropped']} decisões de cases abertos removidas | "
          f"{cens_info['n_in_buffer']} no buffer final ({cens_info['buffer_start']:%Y-%m-%d} em diante, "
          f"fora da avaliação)")
    print(f"Restantes: {len(decisions)} (avaliáveis: {(~decisions['in_buffer']).sum()})")
    print()
    print(decisions["branch"].value_counts())
    print()
    print(decisions.groupby("branch")["cost_seconds"].describe())

    out_path = os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet")
    decisions.to_parquet(out_path)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
