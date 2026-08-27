from pathlib import Path
import json
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

# Permette a Python di trovare il package src/gold_rl.
sys.path.insert(0, str(SRC_DIRECTORY))

# Crea la cartella reports se non esiste.
REPORTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


from gold_rl.data.loader import (
    get_dataset_summary,
    load_market_data,
    print_dataset_summary,
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
            "Il file config.yaml e' vuoto oppure non e' valido."
        )

    if "data" not in config:
        raise KeyError(
            "Nel file config.yaml manca la sezione 'data'."
        )

    return config


def save_summary_to_json(summary: dict) -> Path:
    """Salva il riepilogo del dataset nella cartella reports."""
    output_path = REPORTS_DIRECTORY / "dataset_summary.json"

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

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

    print_dataset_summary(summary)

    print("\nPrime cinque righe")
    print(dataframe.head())

    print("\nUltime cinque righe")
    print(dataframe.tail())

    print("\nTipi delle colonne")
    print(dataframe.dtypes)

    report_path = save_summary_to_json(summary)

    print(
        "\nRiepilogo salvato in:"
        f"\n{report_path.resolve()}"
    )


if __name__ == "__main__":
    main()