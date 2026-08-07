import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS
from src.policies import static_policy, centralized_policy, distance_vector_policy
from src.evaluation import build_counterfactual_estimates, score_policy

pd.set_option("display.width", 200)


def main():
    signal_path = os.path.join(PROCESSED_DIR, "a_validating_decisions_with_cost_signal.parquet")
    decisions = pd.read_parquet(signal_path).sort_values("decision_time").reset_index(drop=True)
    print(f"Decisões carregadas: {len(decisions)}")

    decisions = build_counterfactual_estimates(decisions)

    # Conjunto de avaliação: fora do warm-up e do buffer de censura, ramo realmente
    # roteável e tabela de custo completa (todas as políticas têm o que consultar).
    warmup_end = decisions["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
    known_cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
    eval_mask = (
        (decisions["decision_time"] >= warmup_end)
        & (~decisions["in_buffer"])
        & (decisions["branch"].isin(ROUTABLE_BRANCHES))
        & decisions[known_cols].notna().all(axis=1)
    )
    eval_df = decisions[eval_mask]
    print(f"Conjunto de avaliação (pós warm-up de {WARMUP_DAYS}d, sem buffer, ramo roteável): "
          f"{len(eval_df)}\n")

    policies = {
        "Estática (BPMN fixo)": static_policy(decisions),
        "Centralizada (retreino periódico)": centralized_policy(decisions),
        "Vetor de distância (proposta)": distance_vector_policy(decisions),
    }

    rows = []
    for name, choices in policies.items():
        rows.append(score_policy(eval_df, choices.loc[eval_df.index], name))
    rows.append(score_policy(eval_df, eval_df["branch"], "Log real (decisões observadas)"))
    rows.append(score_policy(eval_df, eval_df["oracle_choice"], "Oráculo (limite inferior)"))

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nDistribuição das escolhas por política:")
    for name, choices in policies.items():
        dist = choices.loc[eval_df.index].value_counts(normalize=True).round(3)
        print(f"  {name}: {dist.to_dict()}")

    out_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "policy_comparison.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
