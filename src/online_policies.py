"""Políticas online avaliadas sobre o DriftEnv, com regret contra o ótimo verdadeiro.

Todas as políticas veem apenas anúncios causalmente disponíveis (casos resolvidos até t).
A cost-tracking usa a MESMA regra do artigo: média móvel de janela W, margem de histerese m
e triggered update (max com beta * tempo decorrido médio dos abertos em delta).
"""

import numpy as np

DAY = 24.0


def _resolved_mean(env, b, t_lo, t_hi):
    """Custo médio dos casos do ramo b resolvidos em [t_lo, t_hi]."""
    res = env.res[b]
    lo = np.searchsorted(res, t_lo, side="left")
    hi = np.searchsorted(res, t_hi, side="right")
    n = hi - lo
    return np.nan if n <= 0 else (env.res_csum[b][hi] - env.res_csum[b][lo]) / n


def _announced_vec(env, t, W, use_tu, delta, beta):
    vals = np.empty(env.k)
    for b in range(env.k):
        T = env.announced(b, t, W)
        if use_tu and not np.isnan(T):
            T = max(T, beta * env.open_elapsed(b, t, delta))
        vals[b] = T
    return vals


def run_policies(env, decision_times, warmup_h,
                 W=1000, m=0.2, use_tu=True, delta_h=14 * DAY, beta=1.0,
                 retrain_h=30 * DAY, central_win_h=60 * DAY):
    """Roda estática, centralizada, cost-tracking e oráculo sobre o mesmo ambiente.
    Retorna dict de arrays de regret por decisão (avaliadas após o warm-up)."""
    dt = np.asarray(decision_times)
    dt = dt[dt >= warmup_h]

    # ---- estática: fixa o ramo de menor custo médio dos resolvidos no warm-up ----
    static_means = np.array([_resolved_mean(env, b, 0.0, warmup_h) for b in range(env.k)])
    static_choice = int(np.nanargmin(static_means))

    # ---- cost-tracking: histerese sobre a tabela corrente ----
    # ---- centralizada: recomputa argmin em lote a cada retrain_h ----
    reg = {p: [] for p in ["static", "centralized", "cost_tracking", "oracle"]}

    ct_cur = None
    central_choice = static_choice
    next_retrain = warmup_h

    idx = {b: b for b in range(env.k)}
    for t in dt:
        truth = env.true_means(t)
        best_true = np.min(truth)

        # centralizada (recomputa argmin em lote na janela deslizante)
        if t >= next_retrain:
            vals = np.array([_resolved_mean(env, b, t - central_win_h, t) for b in range(env.k)])
            if not np.all(np.isnan(vals)):
                central_choice = int(np.nanargmin(vals))
            next_retrain += retrain_h

        # cost-tracking (tabela corrente + histerese)
        tbl = _announced_vec(env, t, W, use_tu, delta_h, beta)
        if np.all(np.isnan(tbl)):
            leader = ct_cur if ct_cur is not None else static_choice
        else:
            leader = int(np.nanargmin(tbl))
            if ct_cur is None or np.isnan(tbl[ct_cur]):
                ct_cur = leader
            elif tbl[leader] < (1 - m) * tbl[ct_cur]:
                ct_cur = leader
            leader = ct_cur

        reg["static"].append(truth[static_choice] - best_true)
        reg["centralized"].append(truth[central_choice] - best_true)
        reg["cost_tracking"].append(truth[leader] - best_true)
        reg["oracle"].append(0.0)

    return {p: np.asarray(v) for p, v in reg.items()}
