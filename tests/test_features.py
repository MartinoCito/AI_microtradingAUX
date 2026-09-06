import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.data.features import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NORMALIZATION_EXCLUDE,
    apply_normalization,
    assert_no_missing_or_infinite,
    create_market_features,
    detect_missing_or_infinite,
    drop_highly_correlated_features,
    fit_normalization_stats,
)


def make_ohlcv_dataframe(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1min")
    trend = 2000.0 + np.linspace(0.0, 12.0, rows)
    wave = 0.8 * np.sin(np.arange(rows) / 7.0)
    close = trend + wave
    open_price = close - 0.15 * np.cos(np.arange(rows) / 5.0)
    high = np.maximum(open_price, close) + 0.25
    low = np.minimum(open_price, close) - 0.25
    volume = 100.0 + (np.arange(rows) % 23) + 0.5 * np.sin(np.arange(rows) / 9.0)

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


def test_selected_feature_columns_exist() -> None:
    dataframe = make_ohlcv_dataframe()
    features = create_market_features(dataframe)

    assert list(features.columns[:-1]) == DEFAULT_FEATURE_COLUMNS
    assert features.index.equals(dataframe.index)


def test_warmup_flag_becomes_true_after_indicator_history() -> None:
    dataframe = make_ohlcv_dataframe(rows=120)
    features = create_market_features(dataframe)

    assert not features["warmup_complete"].iloc[:49].any()
    assert features["warmup_complete"].iloc[80:].all()


def test_missing_and_infinite_checks_ignore_warmup_after_drop() -> None:
    dataframe = make_ohlcv_dataframe()
    features = create_market_features(dataframe, drop_warmup=True)

    issues = detect_missing_or_infinite(features)
    assert issues.empty

    assert_no_missing_or_infinite(features)


def test_training_only_normalization_centers_training_split() -> None:
    dataframe = make_ohlcv_dataframe(rows=260)
    features = create_market_features(dataframe, drop_warmup=True)

    train = features.iloc[:150].copy()
    stats = fit_normalization_stats(train)
    normalized_train = apply_normalization(train, stats)

    normalized_columns = [
        column
        for column in DEFAULT_FEATURE_COLUMNS
        if column not in DEFAULT_NORMALIZATION_EXCLUDE
    ]

    means = normalized_train[normalized_columns].mean().abs()
    stds = normalized_train[normalized_columns].std(ddof=0)

    assert (means < 1e-10).all()
    assert ((stds - 1.0).abs() < 1e-10).all()


def test_correlation_filter_drops_later_duplicate_feature() -> None:
    dataframe = make_ohlcv_dataframe(rows=260)
    features = create_market_features(dataframe, drop_warmup=True)
    features["return_1_clone"] = features["return_1"]

    result = drop_highly_correlated_features(
        train_features=features,
        feature_columns=["return_1", "return_1_clone", "return_4"],
        threshold=0.999999,
        protected_columns=(),
    )

    assert "return_1" in result.kept_columns
    assert "return_1_clone" in result.dropped_columns
    assert result.drop_reasons


def test_strict_feature_contract_rejects_wrong_phase_windows() -> None:
    dataframe = make_ohlcv_dataframe(rows=120)

    with pytest.raises(ValueError):
        create_market_features(dataframe, return_windows=(1, 5, 15))
