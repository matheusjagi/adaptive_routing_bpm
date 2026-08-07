import os

import pandas as pd
import pm4py

from src.config import RAW_LOG_PATH, LOG_PARQUET_PATH


def load_log(use_cache=True):
    if use_cache and os.path.exists(LOG_PARQUET_PATH):
        return pd.read_parquet(LOG_PARQUET_PATH)

    df = pm4py.read_xes(RAW_LOG_PATH)
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(LOG_PARQUET_PATH), exist_ok=True)
    df.to_parquet(LOG_PARQUET_PATH)
    return df
