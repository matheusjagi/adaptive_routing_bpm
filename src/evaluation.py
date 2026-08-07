import numpy as np
import pandas as pd

from src.config import ROUTABLE_BRANCHES, ORACLE_WINDOW_DAYS


def build_counterfactual_estimates(decisions):
    """Estimador contrafactual (SOMENTE para avaliação, nunca para as políticas):
    E[custo | ramo, t] = média do custo realizado das instâncias que seguiram o ramo
    numa janela centrada de ±ORACLE_WINDOW_DAYS em torno de t. Usa informação futura
    de propósito, como proxy de ground truth — limitação declarada no artigo."""
    half_ns = pd.Timedelta(days=ORACLE_WINDOW_DAYS).value
    out = decisions.copy()

    def _to_ns(series):
        return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()

    for branch in ROUTABLE_BRANCHES:
        taken = decisions[decisions["branch"] == branch].sort_values("decision_time")
        times = _to_ns(taken["decision_time"])
        costs = taken["cost_seconds"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(costs)])

        t = _to_ns(out["decision_time"])
        lo = np.searchsorted(times, t - half_ns, side="left")
        hi = np.searchsorted(times, t + half_ns, side="right")
        n = hi - lo
        with np.errstate(invalid="ignore"):
            mean = (csum[hi] - csum[lo]) / n
        mean[n == 0] = np.nan
        out[f"cf_cost_{branch}"] = mean

    cf_cols = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]
    out["oracle_choice"] = out[cf_cols].idxmin(axis=1).str.replace("cf_cost_", "", regex=False)
    return out


def score_policy(evaluation_df, choice_series, name):
    """Custo contrafactual médio da política, taxa de acerto do ramo ótimo e regret
    médio vs. o oráculo (tudo sobre o estimador contrafactual)."""
    df = evaluation_df.loc[choice_series.index].copy()
    df["choice"] = choice_series

    chosen_cost = np.take_along_axis(
        df[[f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]].to_numpy(),
        df["choice"].map({b: i for i, b in enumerate(ROUTABLE_BRANCHES)}).to_numpy()[:, None],
        axis=1,
    ).squeeze(1)
    oracle_cost = df[[f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]].min(axis=1).to_numpy()

    valid = ~np.isnan(chosen_cost) & ~np.isnan(oracle_cost)
    chosen_cost, oracle_cost = chosen_cost[valid], oracle_cost[valid]
    df = df[valid]

    return {
        "policy": name,
        "n": int(valid.sum()),
        "mean_cost_h": chosen_cost.mean() / 3600,
        "median_cost_h": np.median(chosen_cost) / 3600,
        "optimal_rate": float((df["choice"] == df["oracle_choice"]).mean()),
        "mean_regret_h": (chosen_cost - oracle_cost).mean() / 3600,
        "total_regret_days": (chosen_cost - oracle_cost).sum() / 86400,
    }
