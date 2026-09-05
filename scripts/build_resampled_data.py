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

from gold_rl.data.loader import load_market_data
from gold_rl.data.resampling import (
    build_resampling_report,
    resample_ohlcv,
    shift_open_timestamps_to_close,
    validate_resampled_ohlcv,
)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open(mode="r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def main() -> None:
    config = load_config()
    data_config = config["data"]
    resampling_config = config["resampling"]

    processed_directory = PROJECT_ROOT / data_config["processed_directory"]
    processed_directory.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / data_config["raw_path"]

    print(f"Caricamento M1 grezzo: {csv_path.resolve()}")

    m1_dataframe = load_market_data(
        file_path=csv_path,
        separator=data_config["separator"],
        timestamp_column=data_config["timestamp_column"],
        timestamp_format=data_config["timestamp_format"],
        duplicate_policy=data_config["duplicate_policy"],
    )

    if resampling_config["broker_timestamp_convention"] == "open":
        print("Convenzione broker: timestamp di apertura. Applico lo shift a chiusura.")
        m1_shifted = shift_open_timestamps_to_close(
            dataframe=m1_dataframe,
            bar_duration=resampling_config["source_bar_duration"],
        )
    else:
        print("Convenzione broker: timestamp già di chiusura. Nessun shift applicato.")
        m1_shifted = m1_dataframe.copy()

    m1_output_path = processed_directory / "xauusd_m1.parquet"
    m1_shifted.to_parquet(m1_output_path)
    print(f"Salvato M1 (timestamp di chiusura): {m1_output_path.resolve()}")

    m15_dataframe = resample_ohlcv(
        dataframe=m1_shifted,
        target_rule=resampling_config["target_rule"],
        min_coverage_ratio=resampling_config["min_coverage_ratio"],
        source_bar_duration=resampling_config["source_bar_duration"],
    )

    validate_resampled_ohlcv(m15_dataframe)

    m15_output_path = processed_directory / "xauusd_m15.parquet"
    m15_dataframe.to_parquet(m15_output_path)
    print(f"Salvato M15: {m15_output_path.resolve()}")

    report = build_resampling_report(
        source_dataframe=m1_shifted,
        resampled_dataframe=m15_dataframe,
        target_rule=resampling_config["target_rule"],
        min_coverage_ratio=resampling_config["min_coverage_ratio"],
    )

    report_path = REPORTS_DIRECTORY / "resampling_report.json"
    with report_path.open(mode="w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4, default=str)

    print(f"Report di resampling: {report_path.resolve()}")


if __name__ == "__main__":
    main()