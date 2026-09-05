from typing import Any

import pandas as pd


OHLCV_AGGREGATION = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


def shift_open_timestamps_to_close(
    dataframe: pd.DataFrame,
    bar_duration: str = "1min",
) -> pd.DataFrame:
    """
    Sposta i timestamp M1 da 'apertura candela' (convenzione
    MT4/MT5) a 'chiusura candela'.

    Un file M1 MetaTrader etichetta ogni barra con l'orario di
    apertura. Per evitare look-ahead bias nel resampling e nel
    backtest, l'indice viene traslato in avanti della durata
    della barra, cosi' rappresenta l'istante in cui la barra e'
    effettivamente conclusa e disponibile.
    """
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    shift_delta = pd.to_timedelta(bar_duration)

    shifted = dataframe.copy()
    shifted.index = shifted.index + shift_delta
    shifted.index.name = dataframe.index.name

    return shifted


def resample_ohlcv(
    dataframe: pd.DataFrame,
    target_rule: str = "15min",
    min_coverage_ratio: float = 1.0,
    source_bar_duration: str = "1min",
) -> pd.DataFrame:
    """
    Aggrega barre M1 (con timestamp già spostati a chiusura) in
    barre a timeframe superiore (es. M15).

    Ogni barra risultante è etichettata con il proprio istante di
    chiusura (label='right', closed='right'), cosi' contiene solo
    dati M1 gia' disponibili al momento della decisione.

    Le barre con copertura M1 insufficiente (gap, sessioni
    incomplete) vengono scartate secondo min_coverage_ratio.
    """
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    missing_columns = [
        column
        for column in OHLCV_AGGREGATION
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonne OHLCV mancanti per il resampling: {missing_columns}"
        )

    source_delta = pd.to_timedelta(source_bar_duration)
    target_delta = pd.to_timedelta(target_rule)

    expected_bars_per_window = int(
        target_delta / source_delta
    )

    if expected_bars_per_window <= 0:
        raise ValueError(
            "target_rule deve essere un multiplo di "
            "source_bar_duration."
        )

    resampler = dataframe.resample(
        target_rule,
        label="right",
        closed="right",
    )

    resampled = resampler.agg(OHLCV_AGGREGATION)

    bar_counts = resampler["Close"].count()

    coverage_ratio = (
        bar_counts / expected_bars_per_window
    )

    sufficient_coverage_mask = (
        coverage_ratio >= min_coverage_ratio
    )

    resampled = resampled.loc[sufficient_coverage_mask]

    resampled = resampled.dropna(
        subset=list(OHLCV_AGGREGATION)
    )

    resampled["bar_count"] = bar_counts.loc[
        resampled.index
    ].astype("int32")

    resampled["expected_bar_count"] = expected_bars_per_window

    resampled.index.name = dataframe.index.name

    return resampled


def validate_resampled_ohlcv(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verifica la coerenza OHLC delle barre resample-ate, con la
    stessa logica usata per il dataset M1.
    """
    if dataframe.empty:
        raise ValueError("Il dataset resample-ato è vuoto.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    if not dataframe.index.is_monotonic_increasing:
        raise ValueError(
            "Il dataset resample-ato non è ordinato cronologicamente."
        )

    if dataframe.index.has_duplicates:
        raise ValueError(
            "Il dataset resample-ato contiene timestamp duplicati."
        )

    price_columns = ["Open", "High", "Low", "Close"]

    non_positive_price_mask = (
        dataframe[price_columns] <= 0
    ).any(axis=1)

    if non_positive_price_mask.any():
        invalid_count = int(non_positive_price_mask.sum())
        raise ValueError(
            f"Trovate {invalid_count} barre resample-ate con "
            "prezzi nulli o negativi."
        )

    invalid_high_mask = (
        (dataframe["High"] < dataframe["Open"])
        | (dataframe["High"] < dataframe["Close"])
        | (dataframe["High"] < dataframe["Low"])
    )

    invalid_low_mask = (
        (dataframe["Low"] > dataframe["Open"])
        | (dataframe["Low"] > dataframe["Close"])
        | (dataframe["Low"] > dataframe["High"])
    )

    invalid_ohlc_mask = invalid_high_mask | invalid_low_mask

    if invalid_ohlc_mask.any():
        invalid_count = int(invalid_ohlc_mask.sum())
        raise ValueError(
            f"Trovate {invalid_count} barre resample-ate con "
            "OHLC incoerente."
        )

    negative_volume_mask = dataframe["Volume"] < 0

    if negative_volume_mask.any():
        invalid_count = int(negative_volume_mask.sum())
        raise ValueError(
            f"Trovate {invalid_count} barre resample-ate con "
            "volume negativo."
        )


def build_resampling_report(
    source_dataframe: pd.DataFrame,
    resampled_dataframe: pd.DataFrame,
    target_rule: str,
    min_coverage_ratio: float,
) -> dict[str, Any]:
    """
    Costruisce un riepilogo diagnostico del resampling M1 -> M_target.
    """
    total_windows = len(
        source_dataframe.resample(target_rule, label="right", closed="right")
    )

    kept_windows = len(resampled_dataframe)

    dropped_windows = total_windows - kept_windows

    return {
        "source_rows": len(source_dataframe),
        "target_rule": target_rule,
        "min_coverage_ratio": min_coverage_ratio,
        "total_windows": int(total_windows),
        "kept_windows": int(kept_windows),
        "dropped_windows": int(dropped_windows),
        "average_bar_count": float(
            resampled_dataframe["bar_count"].mean()
        ),
        "expected_bar_count": int(
            resampled_dataframe["expected_bar_count"].iloc[0]
        ) if kept_windows > 0 else None,
        "resampled_start": (
            resampled_dataframe.index.min()
            if kept_windows > 0
            else None
        ),
        "resampled_end": (
            resampled_dataframe.index.max()
            if kept_windows > 0
            else None
        ),
    }