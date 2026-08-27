from pathlib import Path
import json
import sys

import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

# Permette a Python di trovare il package src/gold_rl.
sys.path.insert(0, str(SRC_DIRECTORY))

REPORTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

from gold_rl.data.loader import (
    build_data_quality_report,
    get_dataset_summary,
    get_gaps_table,
    load_market_data,
)


def load_config() -> dict:
    """Carica la configurazione principale del progetto."""
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configurazione non trovata: {config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Il file config.yaml è vuoto oppure non è valido."
        )

    if "data" not in config:
        raise KeyError(
            "Nel file config.yaml manca la sezione 'data'."
        )

    return config


def save_data_quality_json(report: dict) -> Path:
    """Salva reports/data_quality.json."""
    output_path = REPORTS_DIRECTORY / "data_quality.json"

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return output_path


def save_gaps_csv(gaps_table: pd.DataFrame) -> Path:
    """Salva reports/gaps.csv con la tabella dei gap."""
    output_path = REPORTS_DIRECTORY / "gaps.csv"

    gaps_table.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def save_data_summary_md(
    summary: dict,
    calendar_report: dict,
) -> Path:
    """
    Salva un riepilogo human-readable in reports/data_summary.md.

    Riassume durata, volume, gap e calendario (weekend/chiusure).
    """
    output_path = REPORTS_DIRECTORY / "data_summary.md"

    lines: list[str] = []

    lines.append("# Data Summary\n")
    lines.append("## Overview\n")
    lines.append(
        f"- Rows: {summary['rows']}\n"
        f"- Start: {summary['start']}\n"
        f"- End: {summary['end']}\n"
        f"- Duration (days): {summary['duration_days']}\n"
        f"- Memory (MB): {summary['memory_mb']}\n"
    )

    lines.append("## Integrity\n")
    lines.append(
        f"- Duplicate timestamps: {summary['duplicate_timestamps']}\n"
        f"- Missing values: {summary['missing_values']}\n"
        f"- Detected gaps: {summary['detected_gaps']}\n"
        f"- Estimated missing bars: {summary['estimated_missing_bars']}\n"
        f"- Largest gap: {summary['largest_gap']}\n"
        f"- Median interval: {summary['median_interval']}\n"
    )

    lines.append("## Volume\n")
    lines.append(
        f"- Zero-volume bars: {summary['zero_volume_bars']}\n"
        f"- Negative-volume bars: {summary['negative_volume_bars']}\n"
        f"- Average volume: {summary['average_volume']}\n"
    )

    lines.append("## Prices\n")
    lines.append(
        f"- Minimum close: {summary['minimum_close']}\n"
        f"- Maximum close: {summary['maximum_close']}\n"
    )

    lines.append("## Calendar\n")
    lines.append(
        f"- Weekend bars: {calendar_report['weekend_bars']}\n"
        f"- Weeks with data: {calendar_report['weeks_with_data']}\n"
        f"- Average trading days per week: {calendar_report['average_trading_days_per_week']}\n"
        f"- Bars per day of week: {calendar_report['bars_per_day_of_week']}\n"
        f"- Long closures (>=2D): {calendar_report['long_closures']}\n"
    )

    lines.append("## Timezone\n")
    lines.append(
        f"- Timezone: {calendar_report['timezone']}\n"
        f"- Note: {calendar_report['timezone_note']}\n"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        output_file.write("\n".join(lines))

    return output_path


def main() -> None:
    config = load_config()
    data_config = config["data"]

    csv_path = PROJECT_ROOT / data_config["raw_path"]

    print(f"Caricamento dataset: {csv_path.resolve()}")

    dataframe = load_market_data(
        file_path=csv_path,
        separator=data_config["separator"],
        timestamp_column=data_config["timestamp_column"],
        timestamp_format=data_config["timestamp_format"],
        duplicate_policy=data_config["duplicate_policy"],
    )

    summary = get_dataset_summary(
        dataframe=dataframe,
        expected_frequency=data_config["expected_frequency"],
    )

    gaps_table = get_gaps_table(
        dataframe=dataframe,
        expected_frequency=data_config["expected_frequency"],
    )

    data_quality_report = build_data_quality_report(
        dataframe=dataframe,
        expected_frequency=data_config["expected_frequency"],
        separator=data_config["separator"],
        timestamp_column=data_config["timestamp_column"],
        timestamp_format=data_config["timestamp_format"],
    )

    calendar_report = data_quality_report["calendar"]

    quality_path = save_data_quality_json(data_quality_report)
    gaps_path = save_gaps_csv(gaps_table)
    summary_md_path = save_data_summary_md(summary, calendar_report)

    print("\nReport generati:")
    print(f"- data_quality.json: {quality_path.resolve()}")
    print(f"- gaps.csv: {gaps_path.resolve()}")
    print(f"- data_summary.md: {summary_md_path.resolve()}")


if __name__ == "__main__":
    main()