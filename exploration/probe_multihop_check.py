"""Reexecuta a comparação de probe_multihop_rip.py (mesma lógica: arestas com >=30
ocorrências, peso = tempo local médio, d(terminal)=0, relaxamento min-plus) e reporta o
SINAL do erro, para conferir a frase 'underestimates ... by about 90%'."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.load_log import load_log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from probe_multihop_rip import build_graph, bellman_ford_propagate, find_cycles

df = load_log()
edges, node_remaining, terminals, all_nodes = build_graph(df)
has_cycle, back = find_cycles(edges, min_edge_count=30)
d, curve, conv, iters = bellman_ford_propagate(edges, terminals, all_nodes, min_edge_count=30)
rows = []
for n in all_nodes:
    p, m = d.get(n, np.inf), node_remaining.get(n, np.nan)
    if np.isinf(p) or np.isnan(m) or m <= 0:
        continue
    rows.append((n, p / 3600, m / 3600, (p - m) / m))
c = pd.DataFrame(rows, columns=["node", "prop_h", "meas_h", "rel"])
print(f"ciclos: {has_cycle} ({len(back)} back-edges) | convergiu: {conv} em {iters} iterações | nós comparados: {len(c)}")
print(f"erro relativo médio |.|: {c['rel'].abs().mean()*100:.1f}% | erro relativo médio (com sinal): {c['rel'].mean()*100:.1f}%")
print(f"erro relativo MEDIANO (com sinal): {c['rel'].median()*100:.1f}% | média com sinal só nós com medido >= 1 h: {c.loc[c['meas_h'] >= 1, 'rel'].mean()*100:.1f}%")
print(f"fração de nós subestimados: {(c['rel'] < 0).mean():.3f} | razão agregada sum(prop)/sum(meas): {c['prop_h'].sum()/c['meas_h'].sum():.3f}")
print(c.sort_values("meas_h", ascending=False).round(2).to_string(index=False))
print("back-edges:", back[:8])
