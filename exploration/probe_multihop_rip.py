"""Sondagem de viabilidade: RIP multi-salto genuíno sobre o grafo do processo BPIC2017.

1. Constrói o grafo direcionado de atividades (nós), com peso de aresta = tempo local
   médio de transição, e destinos = estados terminais (custo-até-o-fim = 0).
2. Analisa loops (ciclos) e alcançabilidade dos terminais.
3. Roda propagação Bellman-Ford (relaxamento iterativo, estilo distance-vector síncrono),
   registrando a curva de convergência e detectando não-convergência por ciclos.
4. Compara o custo-até-o-fim PROPAGADO (multi-salto) vs. o MEDIDO direto (tempo restante
   médio realizado, single-hop) em cada nó, para ver se degrada, empata ou melhora.
"""

import os
import sys
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.load_log import load_log
from src.config import TERMINAL_STATES

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

HOUR = 3600.0


def build_graph(df):
    """Retorna:
       - edges: dict (u -> {v: {'time': mean_local_time_s, 'count': n}})
       - node_remaining: dict node -> tempo restante medio realizado (custo direto medido)
       - terminals: set de nos terminais
    """
    complete = df[df["lifecycle:transition"] == "complete"].copy()
    complete = complete.sort_values(["case:concept:name", "time:timestamp"])

    case_end = complete.groupby("case:concept:name")["time:timestamp"].max()
    complete["case_end"] = complete["case:concept:name"].map(case_end)
    complete["remaining_s"] = (complete["case_end"] - complete["time:timestamp"]).dt.total_seconds()

    complete["next_act"] = complete.groupby("case:concept:name")["concept:name"].shift(-1)
    complete["next_time"] = complete.groupby("case:concept:name")["time:timestamp"].shift(-1)
    complete["local_s"] = (complete["next_time"] - complete["time:timestamp"]).dt.total_seconds()

    # custo direto medido: tempo restante medio por atividade (single-hop / oraculo empirico)
    node_remaining = complete.groupby("concept:name")["remaining_s"].mean().to_dict()

    # arestas: transicoes diretas com tempo local medio
    edges = defaultdict(dict)
    trans = complete.dropna(subset=["next_act"])
    grp = trans.groupby(["concept:name", "next_act"])["local_s"].agg(["mean", "count"])
    for (u, v), row in grp.iterrows():
        edges[u][v] = {"time": row["mean"], "count": int(row["count"])}

    terminals = set(TERMINAL_STATES) & set(complete["concept:name"].unique())

    all_nodes = set(complete["concept:name"].unique())
    return edges, node_remaining, terminals, all_nodes


def find_cycles(edges, min_edge_count=30):
    """Detecta se ha ciclos no grafo (filtrando arestas raras). Retorna (has_cycle, back_edges)."""
    adj = {u: [v for v, d in vs.items() if d["count"] >= min_edge_count]
           for u, vs in edges.items()}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    back_edges = []

    def dfs(u):
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == GRAY:
                back_edges.append((u, v))
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    nodes = set(adj.keys()) | {v for vs in adj.values() for v in vs}
    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return len(back_edges) > 0, back_edges


def reachable_to_terminals(edges, terminals, all_nodes, min_edge_count=30):
    """Quais nos alcancam algum terminal (grafo reverso, BFS a partir dos terminais)."""
    radj = defaultdict(list)
    for u, vs in edges.items():
        for v, d in vs.items():
            if d["count"] >= min_edge_count:
                radj[v].append(u)
    seen = set(terminals)
    stack = list(terminals)
    while stack:
        x = stack.pop()
        for p in radj.get(x, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def bellman_ford_propagate(edges, terminals, all_nodes, min_edge_count=30,
                           max_iters=200, use_split_horizon=False):
    """Propagacao distance-vector sincrona (min-plus). d(terminal)=0.
       d(u) = min_v [ w(u,v) + d(v) ] sobre arestas com contagem suficiente.
       Retorna (d, curva_de_mudanca_por_iter, convergiu, iters)."""
    INF = float("inf")
    d = {n: (0.0 if n in terminals else INF) for n in all_nodes}

    adj = {u: [(v, dd["time"]) for v, dd in vs.items()
               if dd["count"] >= min_edge_count]
           for u, vs in edges.items()}

    curve = []
    converged = False
    for it in range(max_iters):
        new_d = dict(d)
        max_change = 0.0
        for u in all_nodes:
            if u in terminals:
                continue
            best = INF
            for v, w in adj.get(u, []):
                if d[v] < INF:
                    cand = w + d[v]
                    if cand < best:
                        best = cand
            if best < INF:
                if new_d[u] == INF:
                    max_change = INF
                else:
                    max_change = max(max_change, abs(best - new_d[u]))
                new_d[u] = best
        d = new_d
        finite_changes = 0.0 if max_change == INF else max_change
        curve.append(finite_changes if max_change != INF else np.nan)
        if max_change != INF and max_change < 1.0:  # < 1 segundo
            converged = True
            curve = curve[:it + 1]
            break
    iters = len(curve)
    return d, curve, converged, iters


def main():
    print("Carregando log...")
    df = load_log()
    print(f"Eventos: {len(df)} | Cases: {df['case:concept:name'].nunique()}\n")

    edges, node_remaining, terminals, all_nodes = build_graph(df)
    print(f"Nós (atividades): {len(all_nodes)}")
    print(f"Terminais (destinos, custo=0): {sorted(terminals)}")
    print(f"Arestas totais (transições distintas): {sum(len(v) for v in edges.values())}\n")

    # ---------- loops ----------
    has_cycle, back_edges = find_cycles(edges, min_edge_count=30)
    print("=" * 70)
    print("ANÁLISE DE LOOPS (arestas com >= 30 ocorrências)")
    print("=" * 70)
    print(f"Grafo contém ciclos? {'SIM' if has_cycle else 'NÃO'}")
    if has_cycle:
        print(f"Arestas de retorno (back-edges) detectadas: {len(back_edges)}")
        for u, v in back_edges[:12]:
            print(f"    loop: {u}  ->  {v}   (tempo local médio: "
                  f"{edges[u][v]['time']/HOUR:.1f}h, n={edges[u][v]['count']})")
    print("  -> Loops habilitam count-to-infinity: relevante para a narrativa DV.\n")

    # ---------- alcancabilidade ----------
    reach = reachable_to_terminals(edges, terminals, all_nodes, min_edge_count=30)
    unreachable = all_nodes - reach
    print("=" * 70)
    print("ALCANÇABILIDADE DOS TERMINAIS")
    print("=" * 70)
    print(f"Nós que alcançam algum terminal: {len(reach)}/{len(all_nodes)}")
    if unreachable:
        print(f"Nós que NÃO alcançam terminal (custo infinito): {sorted(unreachable)}")
    print()

    # ---------- propagacao ----------
    print("=" * 70)
    print("PROPAGAÇÃO BELLMAN-FORD (distance-vector síncrono)")
    print("=" * 70)
    d, curve, converged, iters = bellman_ford_propagate(edges, terminals, all_nodes,
                                                        min_edge_count=30)
    print(f"Convergiu? {'SIM' if converged else 'NÃO (possível count-to-infinity / oscilação)'}")
    print(f"Iterações até estabilizar (mudança < 1s): {iters}")
    print("Curva de convergência (mudança máxima em horas por iteração, primeiras 15):")
    curve_h = [f"{c/HOUR:.1f}" if not np.isnan(c) else "inf" for c in curve[:15]]
    print("   " + " -> ".join(curve_h))
    print()

    # ---------- comparacao propagado vs medido direto ----------
    print("=" * 70)
    print("CUSTO-ATÉ-O-FIM: PROPAGADO (multi-salto) vs. MEDIDO DIRETO (single-hop)")
    print("=" * 70)
    rows = []
    for n in all_nodes:
        prop = d.get(n, float("inf"))
        direct = node_remaining.get(n, float("nan"))
        if prop == float("inf") or np.isnan(direct):
            continue
        rows.append({
            "node": n,
            "propagado_h": prop / HOUR,
            "medido_h": direct / HOUR,
            "erro_abs_h": (prop - direct) / HOUR,
            "erro_rel": (prop - direct) / direct if direct > 0 else np.nan,
        })
    comp = pd.DataFrame(rows).sort_values("medido_h", ascending=False)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print()
    mae = comp["erro_abs_h"].abs().mean()
    mape = comp["erro_rel"].abs().mean() * 100
    print(f"Erro absoluto médio (propagado vs medido): {mae:.1f}h")
    print(f"Erro relativo médio: {mape:.1f}%")
    print(f"Correlação (propagado, medido): {comp['propagado_h'].corr(comp['medido_h']):.3f}")
    print()
    print("Leitura: erro pequeno => decomposição multi-salto é fiel (viável sem grande")
    print("degradação). Erro grande e sistemático => aditividade/independência quebra")
    print("(multi-salto perderia precisão, mas ainda vale pela narrativa de mecanismo).")


if __name__ == "__main__":
    main()
