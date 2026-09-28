from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.execution import ExecutionCostsConfig
from gold_rl.simulation import simulate_portfolio_over_series

STRATEGIES = ("always_flat", "always_long", "random", "momentum", "ema_crossover")


def load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_test_data(config: dict) -> pd.DataFrame:
    """
    Carica il dataset OHLCV M1 sorgente e ricava esclusivamente il test set
    con lo stesso split cronologico usato dalla Fase 3.

    Il parquet xauusd_m1_test.parquet contiene solo feature normalizzate
    (non OHLCV), quindi non e' una sorgente valida per baseline che devono
    calcolare momentum/EMA sul prezzo.
    """
    data_config = config["data"]
    raw_path = PROJECT_ROOT / data_config["raw_path"]

    from gold_rl.data.loader import load_market_data
    from gold_rl.data.splitter import chronological_date_split

    data = load_market_data(
        file_path=raw_path,
        separator=data_config["separator"],
        timestamp_column=data_config["timestamp_column"],
        timestamp_format=data_config["timestamp_format"],
        duplicate_policy=data_config["duplicate_policy"],
    )

    train_ratio = float(data_config["train_ratio"])
    validation_ratio = float(data_config["validation_ratio"])
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio deve essere compreso tra 0 e 1.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio deve essere compreso tra 0 e 1.")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio deve essere < 1.")

    train_end_index = int(len(data) * train_ratio) - 1
    validation_end_index = int(len(data) * (train_ratio + validation_ratio)) - 1
    if train_end_index < 0 or validation_end_index <= train_end_index:
        raise ValueError("Dataset insufficiente per lo split train/validation/test.")

    train_end = str(data.index[train_end_index])
    validation_end = str(data.index[validation_end_index])

    _, _, test = chronological_date_split(
        dataframe=data,
        train_end=train_end,
        validation_end=validation_end,
    )
    return test

def build_target_positions(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.Series]:
    close = data["Close"].astype(float)
    positions: dict[str, pd.Series] = {
        "always_flat": pd.Series(0, index=data.index, dtype="int8"),
        "always_long": pd.Series(1, index=data.index, dtype="int8"),
    }

    rng = np.random.default_rng(seed)
    positions["random"] = pd.Series(
        rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=len(data)),
        index=data.index,
        dtype="int8",
    )

    # Decisione a t+1: la barra t non può usare il proprio Close per l'ordine a t.
    positions["momentum"] = (
        np.sign(close.diff()).shift(1).fillna(0).astype("int8")
    )

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    positions["ema_crossover"] = (
        np.sign(fast - slow).shift(1).fillna(0).astype("int8")
    )
    return positions


def make_execution_config(environment: dict, with_costs: bool) -> ExecutionCostsConfig:
    if not with_costs:
        return ExecutionCostsConfig(0.0, 0.0, 0.0, 0.0)
    return ExecutionCostsConfig(
        spread_rate=float(environment["spread_rate"]),
        commission_rate=float(environment["commission_rate"]),
        slippage_rate=float(environment["slippage_rate"]),
        turnover_penalty=float(environment["turnover_penalty"]),
    )


def calculate_metrics(
    equity: pd.Series,
    positions: pd.Series,
    initial_balance: float,
) -> dict[str, float]:
    equity = equity.astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualization = np.sqrt(252 * 24 * 60)
    sharpe = (
        float(returns.mean() / volatility * annualization)
        if volatility > 0
        else 0.0
    )
    changes = positions.astype(int).diff().fillna(positions.iloc[0]).abs()
    return {
        "initial_equity": float(initial_balance),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_balance - 1.0),
        "max_drawdown": float(drawdown.min()),
        "volatility_per_bar": volatility,
        "sharpe_annualized_m1": sharpe,
        "trades": int((changes > 0).sum()),
        "turnover": float(changes.sum()),
    }


def run_baselines(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    environment = config["environment"]
    initial_balance = float(environment["initial_balance"])
    seed = int(config["project"]["seed"])
    positions = build_target_positions(data, seed=seed)
    rows: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        target = positions[strategy].to_numpy(dtype=int)
        for cost_mode, with_costs in (
            ("without_costs", False),
            ("with_costs", True),
        ):
            simulation = simulate_portfolio_over_series(
                mid_prices=data["Close"],
                target_positions=target,
                config=make_execution_config(environment, with_costs),
                initial_balance=initial_balance,
            )
            metrics = calculate_metrics(
                simulation["equity"],
                simulation["current_position"],
                initial_balance,
            )
            metrics.update(
                strategy=strategy,
                costs=cost_mode,
                bars=int(len(data)),
                start=str(data.index.min()),
                end=str(data.index.max()),
                seed=seed,
            )
            rows.append(metrics)

    return pd.DataFrame(rows)


def save_equity_figure(data: pd.DataFrame, config: dict) -> None:
    import matplotlib.pyplot as plt

    environment = config["environment"]
    positions = build_target_positions(data, seed=int(config["project"]["seed"]))
    output = PROJECT_ROOT / "reports" / "figures" / "baseline_equity.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for strategy in STRATEGIES:
        target = positions[strategy].to_numpy(dtype=int)
        for ax, with_costs in zip(axes, (False, True)):
            simulation = simulate_portfolio_over_series(
                mid_prices=data["Close"],
                target_positions=target,
                config=make_execution_config(environment, with_costs),
                initial_balance=float(environment["initial_balance"]),
            )
            axes[0 if not with_costs else 1].plot(
                simulation.index,
                simulation["equity"],
                label=strategy,
            )

    axes[0].set_title("Baseline equity — senza costi")
    axes[1].set_title("Baseline equity — con costi")
    for ax in axes:
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    config = load_config()
    data = load_test_data(config)
    report = run_baselines(data, config)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        reports_dir / "baselines.csv",
        index=False,
        float_format="%.12g",
    )
    save_equity_figure(data, config)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
