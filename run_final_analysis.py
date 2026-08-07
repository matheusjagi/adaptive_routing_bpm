"""Pipeline canônico consolidado.

1. Seleção de hiperparâmetros do DV em split temporal: afina no período de tuning
   (mar-jun/2016), congela a melhor configuração e avalia no período de teste
   (jul-dez/2016) — nenhum número final vem de configuração escolhida no próprio teste.
2. Comparação no teste: estática, centralizada retreinada, DV congelado, log real, oráculo.
3. Testes estatísticos pareados por decisão (Wilcoxon e t pareado no regret;
   McNemar na indicadora de escolha ótima).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, ROUTABLE_BRANCHES, WARMUP_DAYS,
    TUNE_TEST_SPLIT, TU_DELTA_DAYS, TU_BETA,
)
from src.cost_table import build_cost_signal
from src.triggered_updates import build_open_case_signal, combine_announcements
from src.policies import static_policy, centralized_policy, hysteresis_policy
from src.evaluation import build_counterfactual_estimates, score_policy

pd.set_option("display.width", 220)

GRID_WINDOWS = [100, 300, 1000]
GRID_MARGINS = [0.0, 0.1, 0.2, 0.3]
GRID_TU = [False, True]

CF_COLS = [f"cf_cost_{b}" for b in ROUTABLE_BRANCHES]
KNOWN_COLS = [f"known_cost_{b}" for b in ROUTABLE_BRANCHES]
BRANCH_IDX = {b: i for i, b in enumerate(ROUTABLE_BRANCHES)}


def regret_per_decision(eval_df, choices):
    cf_mat = eval_df[CF_COLS].to_numpy()
    oracle = np.nanmin(cf_mat, axis=1)
    chosen = np.take_along_axis(
        cf_mat, choices.map(BRANCH_IDX).to_numpy()[:, None], axis=1
    ).squeeze(1)
    return (chosen - oracle) / 3600.0


def mcnemar_exact(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return 1.0, b, c
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return p, b, c


def main():
    base = pd.read_parquet(os.path.join(PROCESSED_DIR, "a_validating_decisions.parquet"))
    base = base.sort_values("decision_time").reset_index(drop=True)

    cf = build_counterfactual_estimates(base)
    cf_keep = cf[CF_COLS + ["oracle_choice"]]
    warmup_end = base["decision_time"].min() + pd.Timedelta(days=WARMUP_DAYS)
    split = pd.Timestamp(TUNE_TEST_SPLIT, tz="UTC")

    print("Pré-computando sinal de casos abertos (triggered updates)...")
    base_with_open = build_open_case_signal(
        build_cost_signal(base, window=GRID_WINDOWS[0]), delta_days=TU_DELTA_DAYS
    )
    open_cols = [f"open_elapsed_{b}" for b in ROUTABLE_BRANCHES]
    open_signal = base_with_open[open_cols]

    def make_signal(window, use_tu):
        signal = build_cost_signal(base, window=window)
        if use_tu:
            signal = signal.join(open_signal)
            signal = combine_announcements(signal, beta=TU_BETA)
        return signal.join(cf_keep)

    def eval_mask(signal, t_lo, t_hi):
        return (
            (signal["decision_time"] >= t_lo)
            & (signal["decision_time"] < t_hi)
            & (~signal["in_buffer"])
            & (signal["branch"].isin(ROUTABLE_BRANCHES))
            & signal[KNOWN_COLS].notna().all(axis=1)
        )

    t_max = base["decision_time"].max() + pd.Timedelta(days=1)

    # ---------- 1. Seleção de hiperparâmetros no período de tuning ----------
    print(f"\n=== Tuning (warm-up até {TUNE_TEST_SPLIT}) ===")
    tune_rows = []
    for window in GRID_WINDOWS:
        for use_tu in GRID_TU:
            signal = make_signal(window, use_tu)
            tune_df = signal[eval_mask(signal, warmup_end, split)]
            for margin in GRID_MARGINS:
                choices = hysteresis_policy(signal, margin)
                s = score_policy(tune_df, choices.loc[tune_df.index],
                                 f"w={window} m={margin} TU={'on' if use_tu else 'off'}")
                s.update({"window": window, "margin": margin, "tu": use_tu})
                tune_rows.append(s)

    tune_results = pd.DataFrame(tune_rows).sort_values("mean_regret_h")
    print(tune_results[["policy", "n", "mean_regret_h", "optimal_rate"]].head(8)
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    best = tune_results.iloc[0]
    print(f"\nConfiguração congelada: janela={best['window']} margem={best['margin']} "
          f"TU={'on' if best['tu'] else 'off'}")

    # ---------- 2. Avaliação congelada no período de teste ----------
    print(f"\n=== Teste ({TUNE_TEST_SPLIT} em diante, configuração congelada) ===")
    signal = make_signal(int(best["window"]), bool(best["tu"]))
    test_df = signal[eval_mask(signal, split, t_max)]
    print(f"Decisões de teste: {len(test_df)}")

    policies = {
        "Estática (BPMN fixo)": static_policy(signal).loc[test_df.index],
        "Centralizada (retreino 30d, janela 60d)": centralized_policy(signal).loc[test_df.index],
        "Vetor de distância (congelado)": hysteresis_policy(signal, float(best["margin"])).loc[test_df.index],
        "Log real (decisões observadas)": test_df["branch"],
        "Oráculo (limite inferior)": test_df["oracle_choice"],
    }

    summary = pd.DataFrame([score_policy(test_df, ch, name) for name, ch in policies.items()])
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ---------- 3. Testes estatísticos pareados ----------
    print("\n=== Testes pareados por decisão (regret; teste unicaudal onde indicado) ===")
    regrets = {name: regret_per_decision(test_df, ch) for name, ch in policies.items()}
    correct = {name: (ch == test_df["oracle_choice"]).to_numpy()
               for name, ch in policies.items()}

    dv = "Vetor de distância (congelado)"
    stat_rows = []
    for rival in ["Estática (BPMN fixo)", "Centralizada (retreino 30d, janela 60d)",
                  "Log real (decisões observadas)"]:
        diff = regrets[rival] - regrets[dv]
        valid = ~np.isnan(diff)
        diff = diff[valid]
        nz = diff[diff != 0]
        if len(nz) > 0:
            w_stat, w_p = stats.wilcoxon(nz, alternative="greater")
        else:
            w_p = 1.0
        t_stat, t_p = stats.ttest_rel(regrets[rival][valid], regrets[dv][valid])
        t_p_onesided = t_p / 2 if t_stat > 0 else 1 - t_p / 2
        mc_p, b, c = mcnemar_exact(correct[dv][valid], correct[rival][valid])
        stat_rows.append({
            "comparação": f"{rival} vs DV",
            "regret_médio_rival_h": diff.mean() + regrets[dv][valid].mean(),
            "regret_médio_dv_h": regrets[dv][valid].mean(),
            "wilcoxon_p (rival pior)": w_p,
            "t_pareado_p (rival pior)": t_p_onesided,
            "mcnemar_p": mc_p,
            "dv_acerta_rival_erra": b,
            "rival_acerta_dv_erra": c,
        })
    stat_results = pd.DataFrame(stat_rows)
    print(stat_results.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---------- 4. Recorte por regime no teste ----------
    print("\n=== Recorte mensal do teste (regret médio, h) ===")
    monthly = test_df.copy()
    monthly["month"] = monthly["decision_time"].dt.to_period("M")
    for name in ["Estática (BPMN fixo)", "Centralizada (retreino 30d, janela 60d)", dv]:
        monthly[name] = regrets[name]
    monthly_tbl = monthly.groupby("month")[
        ["Estática (BPMN fixo)", "Centralizada (retreino 30d, janela 60d)", dv]
    ].mean().round(2)
    print(monthly_tbl.to_string())

    out_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    tune_results.to_csv(os.path.join(out_dir, "final_tuning.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "final_test_comparison.csv"), index=False)
    stat_results.to_csv(os.path.join(out_dir, "final_stat_tests.csv"), index=False)
    monthly_tbl.to_csv(os.path.join(out_dir, "final_monthly_regret.csv"))
    print(f"\nResultados salvos em {out_dir}/final_*.csv")


if __name__ == "__main__":
    main()
