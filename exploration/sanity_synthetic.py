import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from src.synthetic_drift import DriftEnv, DAY
from src.online_policies import run_policies

ANN_RATE = 6.0  # casos/h por ramo -> W=1000 ~ 7 dias de janela


def main():
    horizon = 365 * DAY
    rng = np.random.default_rng(0)
    n_dec = rng.poisson(50 / DAY * horizon)
    dtimes = np.sort(rng.uniform(0, horizon, size=n_dec))
    print(f"Decisões: {len(dtimes)} | W=1000 ~ {1000/ANN_RATE/DAY:.1f} dias de janela\n")

    print(f"{'drift':12s} {'período':>8s}   static  central  cost_tr")
    for drift in ["stationary", "abrupt", "gradual", "oscillating"]:
        for per_d in [180, 90, 45, 20]:
            env = DriftEnv(drift, horizon, amp=100.0, period_h=per_d * DAY,
                           ann_rate_per_h=ANN_RATE, seed=1)
            reg = run_policies(env, dtimes, warmup_h=60 * DAY, W=1000, m=0.2, use_tu=True)
            print(f"{drift:12s} {per_d:6d}d   "
                  f"{np.mean(reg['static']):6.1f}  {np.mean(reg['centralized']):6.1f}  "
                  f"{np.mean(reg['cost_tracking']):6.1f}")
        print()


if __name__ == "__main__":
    main()
