import pandas as pd

from src.config import (
    DECISION_ACTIVITY,
    DECISION_BRANCHES,
    CASE_ATTRS,
    TERMINAL_STATES,
    CENSORING_BUFFER_DAYS,
)


def build_decision_dataset(df):
    """Uma linha por ocorrência de DECISION_ACTIVITY com: timestamp, atributos do caso,
    ramo realmente seguido (next_activity) e custo realizado (tempo até o fim do caso)."""
    complete = df[df["lifecycle:transition"] == "complete"].copy()
    complete = complete.sort_values(["case:concept:name", "time:timestamp"])

    case_end = complete.groupby("case:concept:name")["time:timestamp"].max().rename("case_end")

    complete["next_activity"] = complete.groupby("case:concept:name")["concept:name"].shift(-1)

    decisions = complete[complete["concept:name"] == DECISION_ACTIVITY].copy()
    decisions = decisions[decisions["next_activity"].isin(DECISION_BRANCHES)]
    decisions = decisions.merge(case_end, on="case:concept:name", how="left")

    decisions["cost_seconds"] = (decisions["case_end"] - decisions["time:timestamp"]).dt.total_seconds()

    keep_cols = ["case:concept:name", "time:timestamp", "next_activity", "cost_seconds"] + CASE_ATTRS
    decisions = decisions[keep_cols].rename(columns={
        "case:concept:name": "case_id",
        "time:timestamp": "decision_time",
        "next_activity": "branch",
    })
    return decisions.reset_index(drop=True)


def apply_censoring(decisions, df):
    """Tratamento de censura de fim de log em duas camadas:
    1. Remove decisões de cases sem estado terminal (abertos no fim do log): custo truncado, inválido.
    2. Marca decisões nos últimos CENSORING_BUFFER_DAYS como in_buffer=True: custo verdadeiro
       (o case terminou), podem alimentar o sinal de custo, mas ficam fora da avaliação para
       evitar viés de sobrevivência (cases lentos desse período caíram na camada 1)."""
    complete = df[df["lifecycle:transition"] == "complete"]
    terminal_cases = set(
        complete[complete["concept:name"].isin(TERMINAL_STATES)]["case:concept:name"]
    )
    log_end = df["time:timestamp"].max()

    n_before = len(decisions)
    decisions = decisions[decisions["case_id"].isin(terminal_cases)].copy()
    n_open_dropped = n_before - len(decisions)

    buffer_start = log_end - pd.Timedelta(days=CENSORING_BUFFER_DAYS)
    decisions["in_buffer"] = decisions["decision_time"] > buffer_start

    return decisions.reset_index(drop=True), {
        "n_open_dropped": n_open_dropped,
        "n_in_buffer": int(decisions["in_buffer"].sum()),
        "log_end": log_end,
        "buffer_start": buffer_start,
    }
