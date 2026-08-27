from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
]

NUMERIC_COLUMNS = PRICE_COLUMNS + ["Volume"]


def load_market_data(
    file_path: str | Path,
    separator: str = ";",
    timestamp_column: str = "Date",
    timestamp_format: str = "%Y.%m.%d %H:%M",
    duplicate_policy: str = "keep_last",
) -> pd.DataFrame:
    """
    Carica e valida il dataset OHLCV XAU/USD.

    Parameters
    ----------
    file_path:
        Percorso del file CSV.

    separator:
        Separatore utilizzato nel CSV.

    timestamp_column:
        Nome della colonna temporale.

    timestamp_format:
        Formato utilizzato per interpretare i timestamp.

    duplicate_policy:
        Strategia per i timestamp duplicati.

        Valori possibili:
        - "keep_first"
        - "keep_last"
        - "raise"

    Returns
    -------
    pd.DataFrame
        Dataset ordinato cronologicamente con Date come indice.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"Il percorso indicato non e' un file: {path.resolve()}"
        )

    valid_duplicate_policies = {
        "keep_first",
        "keep_last",
        "raise",
    }

    if duplicate_policy not in valid_duplicate_policies:
        raise ValueError(
            "duplicate_policy deve essere uno tra: "
            f"{sorted(valid_duplicate_policies)}"
        )

    dataframe = pd.read_csv(
        path,
        sep=separator,
        encoding="utf-8",
        skipinitialspace=True,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError("Il file CSV non contiene dati.")

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    validate_required_columns(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
    )

    dataframe = remove_empty_and_html_rows(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
    )

    dataframe[timestamp_column] = pd.to_datetime(
        dataframe[timestamp_column],
        format=timestamp_format,
        errors="coerce",
    )

    invalid_timestamps = dataframe[timestamp_column].isna()

    if invalid_timestamps.any():
        invalid_count = int(invalid_timestamps.sum())

        invalid_examples = dataframe.loc[
            invalid_timestamps
        ].head(5)

        raise ValueError(
            f"Trovati {invalid_count} timestamp non validi.\n"
            f"Prime righe problematiche:\n{invalid_examples}"
        )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid_numeric_rows = dataframe[
        NUMERIC_COLUMNS
    ].isna().any(axis=1)

    if invalid_numeric_rows.any():
        invalid_count = int(invalid_numeric_rows.sum())

        invalid_examples = dataframe.loc[
            invalid_numeric_rows,
            [timestamp_column] + NUMERIC_COLUMNS,
        ].head(5)

        raise ValueError(
            f"Trovate {invalid_count} righe con valori "
            f"OHLCV mancanti o non numerici.\n"
            f"Prime righe problematiche:\n{invalid_examples}"
        )

    dataframe = dataframe.sort_values(
        by=timestamp_column,
        kind="stable",
    )

    dataframe = handle_duplicates(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        duplicate_policy=duplicate_policy,
    )

    dataframe = dataframe.set_index(timestamp_column)
    dataframe.index.name = "Date"

    # Riduce il consumo di memoria.
    dataframe[PRICE_COLUMNS] = dataframe[
        PRICE_COLUMNS
    ].astype("float32")

    dataframe["Volume"] = dataframe["Volume"].astype("float32")

    validate_ohlcv(dataframe)

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> None:
    """
    Controlla che il dataset contenga tutte le colonne richieste.
    """
    expected_columns = [
        timestamp_column,
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Nel dataset mancano le seguenti colonne: "
            f"{missing_columns}. "
            f"Colonne trovate: {list(dataframe.columns)}"
        )


def remove_empty_and_html_rows(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Rimuove righe completamente vuote ed eventuali residui HTML.

    È utile quando il contenuto del CSV è stato copiato da una
    pagina web e contiene stringhe come <br>.
    """
    cleaned = dataframe.dropna(how="all").copy()

    date_as_string = cleaned[timestamp_column].astype(str)

    html_mask = date_as_string.str.contains(
        r"<[^>]+>",
        case=False,
        regex=True,
        na=False,
    )

    if html_mask.any():
        removed_count = int(html_mask.sum())

        print(
            f"Attenzione: rimosse {removed_count} righe "
            "contenenti elementi HTML."
        )

        cleaned = cleaned.loc[~html_mask].copy()

    return cleaned


def handle_duplicates(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    duplicate_policy: str,
) -> pd.DataFrame:
    """
    Gestisce i timestamp duplicati.
    """
    duplicate_mask = dataframe.duplicated(
        subset=[timestamp_column],
        keep=False,
    )

    if not duplicate_mask.any():
        return dataframe

    duplicate_count = int(duplicate_mask.sum())

    duplicate_examples = dataframe.loc[
        duplicate_mask
    ].head(10)

    if duplicate_policy == "raise":
        raise ValueError(
            f"Trovate {duplicate_count} righe con timestamp "
            f"duplicati.\nEsempi:\n{duplicate_examples}"
        )

    keep_value = (
        "first"
        if duplicate_policy == "keep_first"
        else "last"
    )

    print(
        f"Attenzione: trovate {duplicate_count} righe coinvolte "
        f"in duplicati. Strategia applicata: {duplicate_policy}."
    )

    return dataframe.drop_duplicates(
        subset=[timestamp_column],
        keep=keep_value,
    )


def validate_ohlcv(dataframe: pd.DataFrame) -> None:
    """
    Verifica la coerenza delle barre OHLCV.
    """
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    if not dataframe.index.is_monotonic_increasing:
        raise ValueError(
            "Il dataset non è ordinato cronologicamente."
        )

    if dataframe.index.has_duplicates:
        raise ValueError(
            "Il dataset contiene timestamp duplicati."
        )

    non_positive_price_mask = (
        dataframe[PRICE_COLUMNS] <= 0
    ).any(axis=1)

    if non_positive_price_mask.any():
        invalid_count = int(non_positive_price_mask.sum())

        examples = dataframe.loc[
            non_positive_price_mask,
            PRICE_COLUMNS,
        ].head(5)

        raise ValueError(
            f"Trovate {invalid_count} barre con prezzi "
            f"nulli o negativi.\nEsempi:\n{examples}"
        )

    negative_volume_mask = dataframe["Volume"] < 0

    if negative_volume_mask.any():
        invalid_count = int(negative_volume_mask.sum())

        examples = dataframe.loc[
            negative_volume_mask,
            ["Volume"],
        ].head(5)

        raise ValueError(
            f"Trovate {invalid_count} barre con volume "
            f"negativo.\nEsempi:\n{examples}"
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

    invalid_ohlc_mask = (
        invalid_high_mask
        | invalid_low_mask
    )

    if invalid_ohlc_mask.any():
        invalid_count = int(invalid_ohlc_mask.sum())

        examples = dataframe.loc[
            invalid_ohlc_mask,
            PRICE_COLUMNS,
        ].head(5)

        raise ValueError(
            f"Trovate {invalid_count} barre OHLC incoerenti.\n"
            f"Esempi:\n{examples}"
        )


def get_dataset_summary(
    dataframe: pd.DataFrame,
    expected_frequency: str = "1min",
) -> dict[str, Any]:
    """
    Calcola un riepilogo diagnostico del dataset.

    I gap vengono rilevati ma non riempiti.
    """
    if dataframe.empty:
        raise ValueError(
            "Impossibile analizzare un dataset vuoto."
        )

    expected_delta = pd.to_timedelta(expected_frequency)

    time_differences = (
        dataframe.index
        .to_series()
        .diff()
    )

    gap_mask = time_differences > expected_delta
    gaps = time_differences.loc[gap_mask]

    estimated_missing_bars = (
        (gaps / expected_delta) - 1
    ).clip(lower=0)

    memory_bytes = int(
        dataframe.memory_usage(deep=True).sum()
    )

    summary: dict[str, Any] = {
        "rows": len(dataframe),
        "columns": list(dataframe.columns),
        "start": dataframe.index.min(),
        "end": dataframe.index.max(),
        "duration_days": (
            dataframe.index.max() - dataframe.index.min()
        ).days,
        "duplicate_timestamps": int(
            dataframe.index.duplicated().sum()
        ),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "zero_volume_bars": int(
            (dataframe["Volume"] == 0).sum()
        ),
        "negative_volume_bars": int(
            (dataframe["Volume"] < 0).sum()
        ),
        "detected_gaps": int(gap_mask.sum()),
        "estimated_missing_bars": int(
            estimated_missing_bars.sum()
        ),
        "largest_gap": (
            gaps.max()
            if not gaps.empty
            else pd.Timedelta(0)
        ),
        "median_interval": time_differences.median(),
        "memory_mb": round(
            memory_bytes / 1024**2,
            2,
        ),
        "minimum_close": float(
            dataframe["Close"].min()
        ),
        "maximum_close": float(
            dataframe["Close"].max()
        ),
        "average_volume": float(
            dataframe["Volume"].mean()
        ),
    }

    return summary


def get_gaps_table(
    dataframe: pd.DataFrame,
    expected_frequency: str = "1min",
) -> pd.DataFrame:
    """
    Costruisce una tabella con i gap temporali rilevati.

    Ogni riga rappresenta un intervallo in cui mancano una o più barre
    rispetto alla frequenza attesa (es. 1 minuto).
    """
    if dataframe.empty:
        raise ValueError(
            "Impossibile analizzare i gap di un dataset vuoto."
        )

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    expected_delta = pd.to_timedelta(expected_frequency)

    # Differenze tra timestamp consecutivi.
    time_differences = (
        dataframe.index
        .to_series()
        .diff()
    )

    gap_mask = time_differences > expected_delta
    gaps = time_differences.loc[gap_mask]

    if gaps.empty:
        return pd.DataFrame(
            columns=[
                "gap_start",
                "gap_end",
                "gap_duration",
                "missing_bars",
            ]
        )

    rows: list[dict[str, Any]] = []

    # Per ogni gap individuiamo inizio, fine e barre mancanti.
    for gap_timestamp, gap_delta in gaps.items():
        # gap_timestamp è il timestamp di arrivo del gap.
        # Il timestamp precedente è l'ultima barra prima del vuoto.
        previous_timestamp = dataframe.index[
            dataframe.index.get_loc(gap_timestamp) - 1
        ]

        missing_bars = int(
            (gap_delta / expected_delta) - 1
        )

        rows.append(
            {
                "gap_start": previous_timestamp,
                "gap_end": gap_timestamp,
                "gap_duration": gap_delta,
                "missing_bars": missing_bars,
            }
        )

    gaps_table = pd.DataFrame(rows)

    return gaps_table


def analyze_calendar(
    dataframe: pd.DataFrame,
    long_closure_threshold: str = "2D",
) -> dict[str, Any]:
    """
    Analizza il calendario del dataset per individuare weekend
    e chiusure prolungate (es. weekend FX, festività).

    Non assegna automaticamente il fuso orario: fornisce solo
    indicatori diagnostici.
    """
    if dataframe.empty:
        raise ValueError(
            "Impossibile analizzare il calendario di un dataset vuoto."
        )

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "L'indice del DataFrame deve essere un DatetimeIndex."
        )

    # Conteggio barre per giorno della settimana (0=lunedì, 6=domenica).
    day_counts = (
        dataframe.index.dayofweek
        .value_counts()
        .sort_index()
    )

    weekend_bars = int(
        day_counts.get(5, 0) + day_counts.get(6, 0)
    )

    # Numero di settimane con almeno una barra.
    weeks = dataframe.index.to_period("W")
    weeks_with_data = int(
        weeks.value_counts().shape[0]
    )

    # Giorni di trading distinti per settimana (media).
    trading_days_per_week: list[int] = []
    for week, group in dataframe.groupby(weeks):
        unique_days = group.index.dayofweek.unique()
        trading_days_per_week.append(len(unique_days))

    average_trading_days_per_week = float(
        pd.Series(trading_days_per_week).mean()
    )

    # Chiusure prolungate: gap più lunghi di una soglia (es. 2 giorni).
    threshold = pd.to_timedelta(long_closure_threshold)
    expected_delta = pd.to_timedelta("1min")

    time_differences = (
        dataframe.index
        .to_series()
        .diff()
    )
    long_gap_mask = time_differences > threshold
    long_gaps = time_differences.loc[long_gap_mask]

    long_closures: list[dict[str, Any]] = []

    for gap_timestamp, gap_delta in long_gaps.items():
        previous_timestamp = dataframe.index[
            dataframe.index.get_loc(gap_timestamp) - 1
        ]

        missing_bars = int(
            (gap_delta / expected_delta) - 1
        )

        long_closures.append(
            {
                "start": previous_timestamp,
                "end": gap_timestamp,
                "duration": gap_delta,
                "estimated_missing_bars": missing_bars,
            }
        )

    calendar_summary: dict[str, Any] = {
        "bars_per_day_of_week": {
            int(day): int(count)
            for day, count in day_counts.items()
        },
        "weekend_bars": weekend_bars,
        "weeks_with_data": weeks_with_data,
        "average_trading_days_per_week": average_trading_days_per_week,
        "long_closures": long_closures,
        # Nessun fuso orario assegnato automaticamente.
        "timezone": None,
        "timezone_note": (
            "Il fuso orario del broker deve essere determinato da "
            "metadati esterni (es. documentazione del broker, "
            "label EET/EEST, impostazioni MT4/MT5) e NON "
            "assunto automaticamente."
        ),
    }

    return calendar_summary


def build_data_quality_report(
    dataframe: pd.DataFrame,
    expected_frequency: str = "1min",
    separator: str = ";",
    timestamp_column: str = "Date",
    timestamp_format: str = "%Y.%m.%d %H:%M",
) -> dict[str, Any]:
    """
    Costruisce un dizionario compatibile con reports/data_quality.json
    che riassume qualità, calendario e volume del dataset M1.
    """
    summary = get_dataset_summary(
        dataframe=dataframe,
        expected_frequency=expected_frequency,
    )

    calendar = analyze_calendar(dataframe)

    report: dict[str, Any] = {
        "schema": {
            "columns": list(dataframe.columns),
            "separator": separator,
            "timestamp_column": timestamp_column,
            "timestamp_format": timestamp_format,
            # Nessuna timezone assegnata in automatico.
            "timezone": calendar["timezone"],
        },
        "integrity": {
            "rows": summary["rows"],
            "duplicate_timestamps": summary["duplicate_timestamps"],
            "missing_values": summary["missing_values"],
            "detected_gaps": summary["detected_gaps"],
            "estimated_missing_bars": summary["estimated_missing_bars"],
            "largest_gap": summary["largest_gap"],
            "median_interval": summary["median_interval"],
        },
        "volume": {
            "zero_volume_bars": summary["zero_volume_bars"],
            "negative_volume_bars": summary["negative_volume_bars"],
            "average_volume": summary["average_volume"],
        },
        "price": {
            "minimum_close": summary["minimum_close"],
            "maximum_close": summary["maximum_close"],
        },
        "calendar": calendar,
        "notes": {
            "timezone_note": calendar["timezone_note"],
            "frequency_assumption": (
                "Analisi basata su expected_frequency="
                f"{expected_frequency}. I minuti senza "
                "transazioni possono generare gap intraday "
                "legittimi su timeframe M1."
            ),
        },
    }

    return report