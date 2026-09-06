"""
Pipeline Fase 3 - Feature Engineering.

Step:
1. Carica il dataset (M1 raw oppure M_target già resample-ato).
2. Applica lo split cronologico train/validation/test.
3. Genera le feature causali su ciascuno split.
4. Fitta la normalizzazione SOLO sul train (dopo il warm-up).
5. Applica la normalizzazione a train/validation/test.
6. Esegue il filtro di correlazione sul train normalizzato e applica
   lo stesso subset di colonne a validation e test.
7. Salva risultati e artefatti (parquet + json) su disco.
"""
from pathlib import Path
import json
import sys

import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

from gold_rl.data.loader import load_market_data
from gold_rl.data.splitter import chronological_date_split
from gold_rl.data.features import (
    create_market_features,
    fit_normalization_stats,
    apply_normalization,
    drop_highly_correlated_features,
    assert_no_missing_or_infinite,
)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configurazione non trovata: {config_path}")
    with config_path.open(mode="r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml vuoto o non valido.")
    return config


def load_source_dataframe(config: dict) -> pd.DataFrame:
    """
    Carica il dataset sorgente per la Fase 3.

    Usa `data.raw_path` (M1 grezzo) tramite load_market_data. Se in
    futuro si vuole lavorare su un timeframe già aggregato (es. M15),
    sostituire questa funzione con la lettura del relativo parquet in
    data/processed, mantenendo l'indice come DatetimeIndex ordinato.
    """
    data_config = config["data"]
    csv_path = PROJECT_ROOT / data_config["raw_path"]

    print(f"Caricamento dataset sorgente: {csv_path.resolve()}")

    return load_market_data(
        file_path=csv_path,
        separator=data_config["separator"],
        timestamp_column=data_config["timestamp_column"],
        timestamp_format=data_config["timestamp_format"],
        duplicate_policy=data_config["duplicate_policy"],
    )


def compute_split_dates(dataframe: pd.DataFrame, data_config: dict) -> tuple[str, str]:
    """
    Calcola i timestamp di confine train/validation/test in base ai
    ratio configurati in config.yaml (train_ratio, validation_ratio).
    """
    train_ratio = data_config["train_ratio"]
    validation_ratio = data_config["validation_ratio"]

    total_rows = len(dataframe)
    train_end_index = int(total_rows * train_ratio) - 1
    validation_end_index = int(total_rows * (train_ratio + validation_ratio)) - 1

    train_end = dataframe.index[train_end_index]
    validation_end = dataframe.index[validation_end_index]

    return str(train_end), str(validation_end)


def build_features_for_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Genera le feature causali e rimuove le righe di warm-up incompleto."""
    return create_market_features(dataframe, drop_warmup=True)


def save_normalization_stats(stats, output_path: Path) -> None:
    payload = {
        "normalized_columns": list(stats.normalized_columns),
        "excluded_columns": list(stats.excluded_columns),
        "means": stats.means.to_dict(),
        "stds": stats.stds.to_dict(),
    }
    with output_path.open(mode="w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=4, ensure_ascii=False)


def save_correlation_report(result, threshold: float, output_path: Path) -> None:
    payload = {
        "threshold": threshold,
        "kept_columns": list(result.kept_columns),
        "dropped_columns": list(result.dropped_columns),
        "drop_reasons": [
            {"kept": left, "dropped": right, "correlation": value}
            for left, right, value in result.drop_reasons
        ],
    }
    with output_path.open(mode="w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=4, ensure_ascii=False)


def main() -> None:
    config = load_config()
    data_config = config["data"]

    dataframe = load_source_dataframe(config)

    train_end, validation_end = compute_split_dates(dataframe, data_config)
    print(f"Split cronologico -> train_end={train_end} validation_end={validation_end}")

    train_raw, validation_raw, test_raw = chronological_date_split(
        dataframe=dataframe,
        train_end=train_end,
        validation_end=validation_end,
    )

    print("Costruzione feature causali per ciascuno split...")
    train_features = build_features_for_split(train_raw)
    validation_features = build_features_for_split(validation_raw)
    test_features = build_features_for_split(test_raw)

    print("Verifica assenza di missing/infiniti dopo il warm-up...")
    assert_no_missing_or_infinite(train_features)
    assert_no_missing_or_infinite(validation_features)
    assert_no_missing_or_infinite(test_features)

    print("Fit della normalizzazione SOLO sul training set...")
    normalization_stats = fit_normalization_stats(train_features)

    train_normalized = apply_normalization(train_features, normalization_stats)
    validation_normalized = apply_normalization(validation_features, normalization_stats)
    test_normalized = apply_normalization(test_features, normalization_stats)

    print("Filtro delle feature fortemente correlate (calcolato sul train)...")
    correlation_result = drop_highly_correlated_features(
        train_features=train_normalized,
        threshold=0.95,
    )

    kept_columns = list(correlation_result.kept_columns)
    print(f"Feature scartate per correlazione: {correlation_result.dropped_columns}")

    train_final = train_normalized[kept_columns]
    validation_final = validation_normalized[kept_columns]
    test_final = test_normalized[kept_columns]

    processed_directory = PROJECT_ROOT / data_config["processed_directory"]
    processed_directory.mkdir(parents=True, exist_ok=True)

    train_final.to_parquet(processed_directory / "xauusd_m1_train.parquet")
    validation_final.to_parquet(processed_directory / "xauusd_m1_validation.parquet")
    test_final.to_parquet(processed_directory / "xauusd_m1_test.parquet")

    save_normalization_stats(
        normalization_stats,
        REPORTS_DIRECTORY / "normalization_stats.json",
    )
    save_correlation_report(
        correlation_result,
        threshold=0.95,
        output_path=REPORTS_DIRECTORY / "feature_correlation.json",
    )

    print("\nCompletato.")
    print(f"Train: {train_final.shape}, Validation: {validation_final.shape}, Test: {test_final.shape}")
    print(f"Colonne finali: {kept_columns}")


if __name__ == "__main__":
    main()
