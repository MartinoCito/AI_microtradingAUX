from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.data.loader import (
    get_dataset_summary,
    load_market_data,
    print_dataset_summary,
)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configurazione non trovata: {config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    config = load_config()
    data_config = config["data"]

    csv_path = PROJECT_ROOT / data_config["raw_path"]

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


if __name__ == "__main__":
    main()