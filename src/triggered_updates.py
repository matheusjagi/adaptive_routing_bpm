import numpy as np
import pandas as pd

from src.config import ROUTABLE_BRANCHES


def build_open_case_signal(decisions, delta_days=14):
    """Sinal de 'triggered update': para cada instante de decisão t e cada ramo roteável,
    o tempo médio já decorrido dos casos que entraram no ramo nos últimos `delta_days`
    e AINDA NÃO terminaram em t. Sob congestionamento, esse sinal sobe imediatamente,
    sem esperar os casos resolverem (o análogo do triggered update do RIP).
    Usa apenas informação observável em t."""
    decisions = decisions.sort_values("decision_time").reset_index(drop=True)
    delta_ns = pd.Timedelta(days=delta_days).value

    def _to_ns(series):
        return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()

    t_all = _to_ns(decisions["decision_time"])
    out = decisions.copy()

    for branch in ROUTABLE_BRANCHES:
        taken = decisions[decisions["branch"] == branch]
        entry = _to_ns(taken["decision_time"])
        end = _to_ns(taken["case_end"])
        order = np.argsort(entry)
        entry, end = entry[order], end[order]

        t_open = np.full(len(t_all), np.nan)
        for i, t in enumerate(t_all):
            lo = np.searchsorted(entry, t - delta_ns, side="left")
            hi = np.searchsorted(entry, t, side="right")
            if hi <= lo:
                continue
            elapsed = t - entry[lo:hi]
            still_open = end[lo:hi] > t
            if still_open.any():
                t_open[i] = elapsed[still_open].mean() / 1e9  # em segundos

        out[f"open_elapsed_{branch}"] = t_open

    return out


def combine_announcements(decisions, beta=2.0):
    """Anúncio final por ramo: max(custo médio dos resolvidos, beta * tempo decorrido médio
    dos abertos recentes). Em regime estável os dois termos se equivalem e o resolvido domina;
    sob congestionamento o termo dos abertos dispara primeiro."""
    out = decisions.copy()
    for branch in ROUTABLE_BRANCHES:
        resolved = out[f"known_cost_{branch}"]
        open_sig = out[f"open_elapsed_{branch}"] * beta
        out[f"known_cost_{branch}"] = np.fmax(resolved, open_sig)
    return out
