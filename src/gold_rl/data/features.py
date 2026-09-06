'''
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
    '''

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
STATE_COLUMNS = ["current_position", "unrealized_return"]
MARKET_FEATURE_COLUMNS = [
    "return_1",
    "return_4",
    "return_16",
    "atr_over_close",
    "close_ema20_over_atr",
    "close_ema50_over_atr",
    "ema20_ema50_over_atr",
    "candle_body_over_range",
    "upper_wick_over_range",
    "lower_wick_over_range",
    "relative_volume",
    "hour_sin",
    "hour_cos",
]
DEFAULT_FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS + STATE_COLUMNS
DEFAULT_NORMALIZATION_EXCLUDE = (
    "hour_sin",
    "hour_cos",
    "current_position",
    "unrealized_return",
)


@dataclass(frozen=True)
class NormalizationStats:
    means: pd.Series
    stds: pd.Series
    normalized_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]


@dataclass(frozen=True)
class CorrelationFilterResult:
    kept_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    drop_reasons: tuple[tuple[str, str, float], ...]


def _validate_input_dataframe(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError("L'indice deve essere un DatetimeIndex.")

    if not dataframe.index.is_monotonic_increasing:
        raise ValueError("Il dataset deve essere ordinato cronologicamente.")

    missing_columns = [
        column for column in OHLCV_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Colonne OHLCV mancanti: {missing_columns}"
        )


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype("float64").ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1.0 / window,
        adjust=False,
        min_periods=window,
    ).mean()


def build_state_frame(
    index: pd.DatetimeIndex,
    state_frame: pd.DataFrame | None = None,
    default_position: float = 0.0,
    default_unrealized_return: float = 0.0,
) -> pd.DataFrame:
    if state_frame is None:
        return pd.DataFrame(
            {
                "current_position": default_position,
                "unrealized_return": default_unrealized_return,
            },
            index=index,
            dtype="float64",
        )

    aligned = state_frame.reindex(index).copy()

    missing_fraction = aligned[["current_position", "unrealized_return"]].isna().mean().max()
    if missing_fraction > 0.01:
        print(
            f"[warn] reindex di state_frame ha introdotto NaN su "
            f"{missing_fraction:.1%} delle righe: verifica l'allineamento "
            "temporale tra OHLCV ed environment RL."
        )

    for column, default_value in {
        "current_position": default_position,
        "unrealized_return": default_unrealized_return,
    }.items():
        if column not in aligned.columns:
            aligned[column] = default_value
        aligned[column] = aligned[column].fillna(default_value).astype("float64")

    return aligned[["current_position", "unrealized_return"]]


def create_market_features(
    dataframe: pd.DataFrame,
    state_frame: pd.DataFrame | None = None,
    return_windows: Sequence[int] = (1, 4, 16),
    atr_window: int = 14,
    ema_windows: Sequence[int] = (20, 50),
    relative_volume_window: int = 20,
    drop_warmup: bool = False,
    strict_windows: bool = True
) -> pd.DataFrame:
    """
    Genera le feature causali della Fase 3.

    Il warm-up complessivo (colonna `warmup_complete`) e' dominato da EMA50
    (min_periods=50), non da relative_volume (min_periods=20) ne' da ATR
    (min_periods=14). Con i default correnti, warmup_complete diventa True
    dalla 50-esima barra osservata (indice 0-based: riga 49).
    """
    _validate_input_dataframe(dataframe)

    if strict_windows:
        if tuple(return_windows) != (1, 4, 16):
            raise ValueError(
                "return_windows deve essere (1, 4, 16) in modalità strict. "
                "Passa strict_windows=False per sperimentare altre finestre."
            )
        if tuple(ema_windows) != (20, 50):
            raise ValueError(
                "ema_windows deve essere (20, 50) in modalità strict. "
                "Passa strict_windows=False per sperimentare altre finestre."
            )
    else:
        print(
            f"[warn] finestre non standard rispetto alla Fase 3: "
            f"return_windows={tuple(return_windows)} ema_windows={tuple(ema_windows)}"
        )

    features = pd.DataFrame(index=dataframe.index)

    open_price = dataframe["Open"].astype("float64")
    high = dataframe["High"].astype("float64")
    low = dataframe["Low"].astype("float64")
    close = dataframe["Close"].astype("float64")
    open_price = dataframe["Open"].astype("float64")
    volume = dataframe["Volume"].astype("float64")

    invalid_bars = high < low
    if invalid_bars.any():
        invalid_count = int(invalid_bars.sum())
        raise ValueError(
            f"Trovate {invalid_count} barre con High < Low: i dati non sono "
            "stati validati a monte da load_market_data/validate_ohlcv."
        )

    atr = calculate_atr(high=high, low=low, close=close, window=atr_window)
    ema20 = calculate_ema(close, span=ema_windows[0])
    ema50 = calculate_ema(close, span=ema_windows[1])
    

    bar_range = (high - low).replace(0.0, np.nan)
    safe_atr = atr.replace(0.0, np.nan)

    features["return_1"] = close.pct_change(1)
    features["return_4"] = close.pct_change(4)
    features["return_16"] = close.pct_change(16)
    features["atr_over_close"] = atr / close.replace(0.0, np.nan)
    features["close_ema20_over_atr"] = (close - ema20) / safe_atr
    features["close_ema50_over_atr"] = (close - ema50) / safe_atr
    features["ema20_ema50_over_atr"] = (ema20 - ema50) / safe_atr
    features["candle_body_over_range"] = (close - open_price) / bar_range
    features["upper_wick_over_range"] = (
        high - np.maximum(open_price, close)
    ) / bar_range
    features["lower_wick_over_range"] = (
        np.minimum(open_price, close) - low
    ) / bar_range
    features["relative_volume"] = volume / volume.rolling(
        window=relative_volume_window,
        min_periods=relative_volume_window,
    ).mean()

    minute_of_day = dataframe.index.hour * 60 + dataframe.index.minute
    features["hour_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)

    states = build_state_frame(index=dataframe.index, state_frame=state_frame)
    features[STATE_COLUMNS] = states[STATE_COLUMNS]

    features["warmup_complete"] = features[MARKET_FEATURE_COLUMNS].notna().all(axis=1)

    ordered_columns = DEFAULT_FEATURE_COLUMNS + ["warmup_complete"]
    features = features[ordered_columns]

    if drop_warmup:
        features = features.loc[features["warmup_complete"]].copy()

    return features


build_feature_frame = create_market_features


def detect_missing_or_infinite(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    subset = dataframe[columns]

    rows: list[dict[str, object]] = []
    for column in columns:
        series = subset[column]
        missing_mask = series.isna()
        infinite_mask = np.isinf(series.to_numpy(dtype="float64", copy=False))

        if missing_mask.any() or infinite_mask.any():
            rows.append(
                {
                    "feature": column,
                    "missing_count": int(missing_mask.sum()),
                    "infinite_count": int(infinite_mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def assert_no_missing_or_infinite(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> None:
    issues = detect_missing_or_infinite(dataframe, feature_columns=feature_columns)
    if not issues.empty:
        raise ValueError(
            "Trovati valori mancanti o infiniti nelle feature:\n"
            f"{issues.to_string(index=False)}"
        )


def fit_normalization_stats(
    train_features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    exclude_columns: Iterable[str] = DEFAULT_NORMALIZATION_EXCLUDE,
) -> NormalizationStats:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    excluded = tuple(column for column in exclude_columns if column in columns)
    normalized_columns = tuple(
        column for column in columns if column not in excluded
    )

    if not normalized_columns:
        raise ValueError("Nessuna colonna disponibile per la normalizzazione.")

    reference = train_features.loc[:, normalized_columns]
    assert_no_missing_or_infinite(reference, feature_columns=normalized_columns)

    means = reference.mean(axis=0)
    stds = reference.std(axis=0, ddof=0).replace(0.0, 1.0)

    return NormalizationStats(
        means=means,
        stds=stds,
        normalized_columns=normalized_columns,
        excluded_columns=excluded,
    )


def apply_normalization(
    features: pd.DataFrame,
    stats: NormalizationStats,
) -> pd.DataFrame:
    normalized = features.copy()
    normalized.loc[:, stats.normalized_columns] = (
        normalized.loc[:, stats.normalized_columns] - stats.means
    ) / stats.stds
    return normalized


def drop_highly_correlated_features(
    train_features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    threshold: float = 0.95,
    protected_columns: Iterable[str] = (
        "hour_sin",
        "hour_cos",
        "current_position",
        "unrealized_return",
    ),
) -> CorrelationFilterResult:
    """
    Rimuove feature con correlazione di Pearson (lineare) >= threshold.

    Limite noto: non rileva dipendenze non lineari (es. atr_over_close vs
    relative_volume in regimi di alta volatilità). Estensione futura
    possibile con correlazione di Spearman o mutual information.
    """
    ordered_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    protected = set(protected_columns)

    candidate_columns = [
        column for column in ordered_columns if column not in protected
    ]

    if len(candidate_columns) < 2:
        return CorrelationFilterResult(
            kept_columns=tuple(ordered_columns),
            dropped_columns=tuple(),
            drop_reasons=tuple(),
        )

    correlation_matrix = train_features.loc[:, candidate_columns].corr().abs()

    dropped: set[str] = set()
    reasons: list[tuple[str, str, float]] = []

    for left_index, left_column in enumerate(candidate_columns):
        if left_column in dropped:
            continue

        for right_column in candidate_columns[left_index + 1 :]:
            if right_column in dropped:
                continue

            correlation_value = correlation_matrix.loc[left_column, right_column]
            if pd.isna(correlation_value):
                continue

            if correlation_value >= threshold:
                dropped.add(right_column)
                reasons.append(
                    (left_column, right_column, float(correlation_value))
                )

    kept_columns = tuple(
        column for column in ordered_columns if column not in dropped
    )

    return CorrelationFilterResult(
        kept_columns=kept_columns,
        dropped_columns=tuple(column for column in ordered_columns if column in dropped),
        drop_reasons=tuple(reasons),
    )
