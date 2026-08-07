import numpy as np
import pandas as pd

from src.config import DECISION_BRANCHES


def build_cost_signal(decisions, window=30, statistic="mean"):
    """Para cada instância de decisão (em decision_time), o custo 'conhecido' de cada
    ramo nesse instante = média (ou mediana) móvel das últimas `window` instâncias do ramo
    já RESOLVIDAS (case_end <= decision_time). Nunca usa informação futura."""
    decisions = decisions.sort_values("decision_time").reset_index(drop=True)
    case_end = decisions["decision_time"] + pd.to_timedelta(decisions["cost_seconds"], unit="s")
    decisions["case_end"] = case_end.astype(decisions["decision_time"].dtype)

    out = decisions.copy()  # já ordenado por decision_time, índice 0..N-1 nessa ordem

    for branch in DECISION_BRANCHES:
        branch_df = decisions[decisions["branch"] == branch][["case_end", "cost_seconds"]].copy()
        branch_df = branch_df.sort_values("case_end").reset_index(drop=True)
        rolling = branch_df["cost_seconds"].rolling(window=window, min_periods=1)
        branch_df["known_cost"] = rolling.median() if statistic == "median" else rolling.mean()
        branch_df["known_cost"] = branch_df["known_cost"].shift(1)  # só pode usar resolvidos ANTES deste

        lookup = branch_df[["case_end", "known_cost"]].dropna().sort_values("case_end")
        lookup = lookup.rename(columns={"case_end": "decision_time"})

        merged = pd.merge_asof(
            out[["decision_time"]],
            lookup,
            on="decision_time",
            direction="backward",
        )
        out[f"known_cost_{branch}"] = merged["known_cost"].values

    return out


def build_cost_signal_time(decisions, window_days=14):
    """Variante com janela TEMPORAL (a tradução fiel do protocolo: anúncios em intervalos
    de tempo, não a cada N casos): custo conhecido de um ramo em t = média dos casos do
    ramo resolvidos em [t - window_days, t]. Elimina a defasagem assimétrica entre ramos
    de volumes diferentes que a janela por contagem introduz."""
    decisions = decisions.sort_values("decision_time").reset_index(drop=True)
    case_end = decisions["decision_time"] + pd.to_timedelta(decisions["cost_seconds"], unit="s")
    decisions["case_end"] = case_end.astype(decisions["decision_time"].dtype)

    window_ns = pd.Timedelta(days=window_days).value

    def _to_ns(series):
        return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()

    t_all = _to_ns(decisions["decision_time"])
    out = decisions.copy()

    for branch in DECISION_BRANCHES:
        branch_df = decisions[decisions["branch"] == branch].sort_values("case_end")
        ends = _to_ns(branch_df["case_end"])
        costs = branch_df["cost_seconds"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(costs)])

        lo = np.searchsorted(ends, t_all - window_ns, side="left")
        hi = np.searchsorted(ends, t_all, side="left")  # estritamente antes de t
        n = hi - lo
        with np.errstate(invalid="ignore"):
            mean = (csum[hi] - csum[lo]) / n
        mean[n == 0] = np.nan
        out[f"known_cost_{branch}"] = mean

    return out
