import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.data.features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NORMALIZATION_EXCLUDE,
    apply_normalization,
    create_market_features,
    fit_normalization_stats,
)
from gold_rl.data.leakage_checks import (
    assert_future_append_invariant,
    assert_past_normalization_invariant,
)


def make_ohlcv_dataframe(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=rows, freq="1min")
    base = 2050.0 + np.linspace(0.0, 18.0, rows)
    seasonal = 0.9 * np.sin(np.arange(rows) / 8.0)
    close = base + seasonal
    open_price = close - 0.12 * np.cos(np.arange(rows) / 6.0)
    high = np.maximum(open_price, close) + 0.2
    low = np.minimum(open_price, close) - 0.2
    volume = 120.0 + (np.arange(rows) % 17) + 0.25 * np.cos(np.arange(rows) / 10.0)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_future_append_does_not_change_past_features() -> None:
    dataframe = make_ohlcv_dataframe(rows=260)
    history = dataframe.iloc[:180].copy()
    future = dataframe.iloc[180:].copy()

    assert_future_append_invariant(
        feature_builder=lambda frame: create_market_features(frame),
        history_dataframe=history,
        future_dataframe=future,
    )


def test_future_append_does_not_change_past_normalized_values_when_train_stats_are_reused() -> None:
    dataframe = make_ohlcv_dataframe(rows=260)
    history = dataframe.iloc[:180].copy()
    future = dataframe.iloc[180:].copy()

    past_features = create_market_features(history, drop_warmup=True)
    appended_features = create_market_features(dataframe, drop_warmup=True)

    train_past = past_features.iloc[:100].copy()
    stats = fit_normalization_stats(train_past)

    normalized_past = apply_normalization(past_features, stats)
    normalized_appended = apply_normalization(appended_features, stats)

    compare_columns = [
        column
        for column in DEFAULT_FEATURE_COLUMNS
        if column not in DEFAULT_NORMALIZATION_EXCLUDE
    ]

    assert_past_normalization_invariant(
        past_features=past_features,
        appended_features=appended_features,
        normalized_past=normalized_past,
        normalized_appended=normalized_appended,
        feature_columns=compare_columns,
    )
