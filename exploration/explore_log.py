import os

import pandas as pd
import pm4py

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "BPI_Challenge_2017.xes.gz")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load():
    return pm4py.read_xes(LOG_PATH)


def directly_follows_branching(df):
    """Para cada atividade (eventos 'complete'), distribuição da atividade seguinte na mesma trace."""
    complete = df[df["lifecycle:transition"] == "complete"].copy()
    complete = complete.sort_values(["case:concept:name", "time:timestamp"])
    complete["next_activity"] = complete.groupby("case:concept:name")["concept:name"].shift(-1)

    branching = (
        complete.dropna(subset=["next_activity"])
        .groupby(["concept:name", "next_activity"])
        .size()
        .reset_index(name="count")
    )
    return branching


def report_branching(branching, min_count=200):
    print("=" * 80)
    print("PONTOS COM RAMIFICAÇÃO REAL (>=2 atividades seguintes possíveis, suporte >= %d)" % min_count)
    print("=" * 80)
    totals = branching.groupby("concept:name")["count"].sum().rename("total")
    branching = branching.merge(totals, on="concept:name")
    branching["pct"] = branching["count"] / branching["total"]

    for activity, group in branching.groupby("concept:name"):
        group = group[group["count"] >= min_count].sort_values("count", ascending=False)
        if len(group) < 2:
            continue
        print(f"\n{activity}  (total={int(group['total'].iloc[0])})")
        for _, row in group.iterrows():
            print(f"    -> {row['next_activity']:30s} count={int(row['count']):6d}  ({row['pct']*100:5.1f}%)")


def queue_dynamics(df, activity_prefix="W_"):
    """Tempo de espera (schedule->start) e processamento (start->complete) por atividade W_*."""
    w = df[df["concept:name"].str.startswith(activity_prefix)].copy()
    w = w.sort_values(["case:concept:name", "concept:name", "time:timestamp"])

    print("\n" + "=" * 80)
    print(f"DINÂMICA DE FILA — atividades '{activity_prefix}*' (lifecycle completo)")
    print("=" * 80)
    print(w["concept:name"].value_counts())

    # Para cada instância de workitem (EventID compartilhado entre schedule/start/complete não é garantido;
    # workitems têm EventID próprio por linha). Vamos usar EventOrigin=Workflow e parear schedule/start/complete
    # dentro da mesma atividade+case, em ordem temporal, como aproximação (pareamento sequencial).
    records = []
    for (case, act), g in w.groupby(["case:concept:name", "concept:name"]):
        g = g.sort_values("time:timestamp")
        sched = g[g["lifecycle:transition"] == "schedule"]["time:timestamp"].tolist()
        start = g[g["lifecycle:transition"] == "start"]["time:timestamp"].tolist()
        comp = g[g["lifecycle:transition"] == "complete"]["time:timestamp"].tolist()
        for s, st, c in zip(sched, start, comp):
            records.append({
                "activity": act,
                "case": case,
                "wait_seconds": (st - s).total_seconds(),
                "process_seconds": (c - st).total_seconds(),
            })

    qd = pd.DataFrame(records)
    if qd.empty:
        print("Sem registros pareados schedule/start/complete.")
        return qd

    qd = qd[(qd["wait_seconds"] >= 0) & (qd["process_seconds"] >= 0)]
    summary = qd.groupby("activity")[["wait_seconds", "process_seconds"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    print("\nEspera (schedule->start) e processamento (start->complete), em segundos:")
    print(summary)
    return qd


def queue_load_over_time(qd, activity, freq="W"):
    """Série temporal: nº de tarefas em espera/concorrentes por janela, para 1 atividade W_."""
    sub = qd[qd["activity"] == activity]
    print(f"\nDistribuição semanal do nº de instâncias completadas — {activity}:")
    # placeholder: contagem simples por semana usando wait_seconds index não é direto sem timestamp absoluto aqui
    print(sub.describe())


def main():
    print("Carregando log...")
    df = load()
    print(f"Eventos: {len(df)} | Cases: {df['case:concept:name'].nunique()}\n")

    branching = directly_follows_branching(df)
    report_branching(branching, min_count=200)

    qd = queue_dynamics(df, "W_")

    out_path = os.path.join(_PROJECT_ROOT, "data", "processed", "queue_dynamics_sample.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    qd.to_csv(out_path, index=False)
    print(f"\nAmostra salva em {out_path}")


if __name__ == "__main__":
    main()
