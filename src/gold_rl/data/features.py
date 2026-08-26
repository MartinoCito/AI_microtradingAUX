from collections.abc import Sequence

import numpy as np
import pandas as pd


OHLCV_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def calculate_rsi(
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calcola il Relative Strength Index."""
    price_change = close.diff()

    gains = price_change.clip(lower=0.0)
    losses = -price_change.clip(upper=0.0)

    average_gain = gains.ewm(
        alpha=1.0 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    average_loss = losses.ewm(
        alpha=1.0 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0.0, np.nan)
    )

    rsi = 100.0 - (
        100.0 / (1.0 + relative_strength)
    )

    no_movement = (
        (average_gain == 0.0)
        & (average_loss == 0.0)
    )

    rsi = rsi.mask(no_movement, 50.0)

    rsi = rsi.mask(
        (average_gain > 0.0)
        & (average_loss == 0.0),
        100.0,
    )

    return rsi


def create_market_features(
    dataframe: pd.DataFrame,
    return_windows: Sequence[int],
    volatility_windows: Sequence[int],
    moving_average_windows: Sequence[int],
    rsi_window: int = 14,
    gap_threshold_minutes: float = 5.0,
) -> pd.DataFrame:
    """
    Genera le feature usando esclusivamente presente e passato.
    """
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    missing_columns = [
        column
        for column in OHLCV_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonne OHLCV mancanti: {missing_columns}"
        )

    features = dataframe.copy()

    close = features["Close"].astype("float64")
    open_price = features["Open"].astype("float64")
    high = features["High"].astype("float64")
    low = features["Low"].astype("float64")
    volume = features["Volume"].astype("float64")

    time_delta_minutes = (
        features.index
        .to_series()
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    features["time_delta_minutes"] = time_delta_minutes

    features["is_gap"] = (
        time_delta_minutes > gap_threshold_minutes
    ).astype("float32")

    features["log_return_1"] = np.log(
        close / close.shift(1)
    )

    for window in return_windows:
        features[f"log_return_{window}"] = np.log(
            close / close.shift(window)
        )

    for window in volatility_windows:
        features[f"volatility_{window}"] = (
            features["log_return_1"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
        )

    for window in moving_average_windows:
        moving_average = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

        features[f"ma_distance_{window}"] = (
            close / moving_average
        ) - 1.0

    features["bar_return"] = (
        close / open_price
    ) - 1.0

    features["high_low_range"] = (
        high - low
    ) / close

    features["upper_wick"] = (
        high - np.maximum(open_price, close)
    ) / close

    features["lower_wick"] = (
        np.minimum(open_price, close) - low
    ) / close

    bar_range = high - low

    features["close_location"] = (
        (close - low)
        / bar_range.replace(0.0, np.nan)
    ).fillna(0.5)

    features["rsi_14"] = (
        calculate_rsi(
            close=close,
            window=rsi_window,
        )
        / 100.0
    )

    features["log_volume"] = np.log1p(volume)

    features["log_volume_change"] = (
        features["log_volume"].diff()
    )

    minute_of_day = (
        features.index.hour * 60
        + features.index.minute
    )

    features["time_sin"] = np.sin(
        2.0 * np.pi * minute_of_day / 1440.0
    )

    features["time_cos"] = np.cos(
        2.0 * np.pi * minute_of_day / 1440.0
    )

    day_of_week = features.index.dayofweek

    features["weekday_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 7.0
    )

    features["weekday_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 7.0
    )
    