import numpy as np
import pandas as pd

from src.config import ROUTABLE_BRANCHES, WARMUP_DAYS, RETRAIN_DAYS


def _resolved_mean_costs(decisions, cutoff, window_days=None):
    """Custo médio por ramo roteável usando apenas cases RESOLVIDOS até `cutoff`
    (case_end <= cutoff) — mesma restrição de causalidade do sinal de custo.
    Se `window_days` for dado, considera só os resolvidos nessa janela antes do cutoff."""
    resolved = decisions[decisions["case_end"] <= cutoff]
    if window_days is not None:
        resolved = resolved[resolved["case_end"] > cutoff - pd.Timedelta(days=window_days)]
    resolved = resolved[resolved["branch"].isin(ROUTABLE_BRANCHES)]
    means = resolved.groupby("branch")["cost_seconds"].mean()
    return means.reindex(ROUTABLE_BRANCHES)


def static_policy(decisions):
    """Baseline estática: fixa, no fim do warm-up, o ramo de menor custo médio
    observado até ali e nunca mais muda (o gateway XOR tradicional de um BPMN estático)."""
    start = decisions["decision_time"].min()
    cutoff = start + pd.Timedelta(days=WARMUP_DAYS)
    means = _resolved_mean_costs(decisions, cutoff)
    choice = means.idxmin()
    return pd.Series(choice, index=decisions.index, name="static_choice")


def centralized_policy(decisions, window_days=60):
    """Baseline centralizada retreinada: a cada RETRAIN_DAYS recalcula em batch o custo
    médio por ramo sobre os cases resolvidos na janela recente (`window_days`) e fixa o
    ramo mais barato até o próximo retreino (mineração/otimização periódica centralizada)."""
    decisions = decisions.sort_values("decision_time")
    start = decisions["decision_time"].min()
    end = decisions["decision_time"].max()

    choices = pd.Series(index=decisions.index, dtype=object, name="centralized_choice")
    boundary = start + pd.Timedelta(days=WARMUP_DAYS)
    current_choice = None

    while boundary <= end + pd.Timedelta(days=RETRAIN_DAYS):
        means = _resolved_mean_costs(decisions, boundary, window_days=window_days)
        if means.notna().any():
            current_choice = means.idxmin()
        window_end = boundary + pd.Timedelta(days=RETRAIN_DAYS)
        mask = (decisions["decision_time"] >= boundary) & (decisions["decision_time"] < window_end)
        choices.loc[decisions.index[mask]] = current_choice
        boundary = window_end

    return choices


def distance_vector_policy(decisions):
    """Política proposta: a cada decisão consulta a tabela de custo corrente
    (known_cost_*, atualizada continuamente a cada case resolvido — o 'anúncio'
    do vizinho) e escolhe o ramo roteável de menor custo anunciado."""
    cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
    known = decisions[cols].rename(columns=lambda c: c.replace("known_cost_", ""))
    has_table = known.notna().any(axis=1)
    choices = pd.Series(pd.NA, index=decisions.index, dtype=object, name="dv_choice")
    choices[has_table] = known[has_table].idxmin(axis=1)
    return choices


def hysteresis_policy(decisions, margin):
    """Vetor de distância com hold-down/histerese: só troca de rota quando o novo líder
    é melhor que a rota corrente por pelo menos `margin` (fração relativa) — o análogo
    de route dampening em protocolos de roteamento."""
    cols = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
    known = decisions[cols].to_numpy()

    choices = np.full(len(decisions), -1, dtype=int)
    current = -1
    for i in range(len(decisions)):
        row = known[i]
        if np.isnan(row).all():
            continue
        best = np.nanargmin(row)
        if current == -1 or np.isnan(row[current]):
            current = best
        elif row[best] < row[current] * (1.0 - margin):
            current = best
        choices[i] = current

    return pd.Series(
        [ROUTABLE_BRANCHES[c] if c >= 0 else pd.NA for c in choices],
        index=decisions.index, dtype=object,
    )
